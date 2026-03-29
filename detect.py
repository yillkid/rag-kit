"""
RAG Vision — 地標辨識腳本

讀取 Google Sheet 地標資料 + 圖片 → 呼叫 Gemini API → 辨識地點

使用方式：
  # 辨識單張圖片
  python detect.py photo.jpg

  # 辨識多張圖片
  python detect.py photo1.jpg photo2.jpg photo3.jpg

  # 辨識資料夾內所有圖片
  python detect.py images/

  # 從 URL 辨識
  python detect.py https://hackmd.io/_uploads/B1Yu1w8s-g.png

  # 從 HackMD 頁面批次辨識（自動抓取頁面內所有圖片）
  python detect.py --hackmd https://hackmd.io/@yillkid/Hy42AcLj-g

  # 指定 API Key
  python detect.py --key AIzaSy... photo.jpg

環境變數：
  GEMINI_API_KEY=你的金鑰
"""

import argparse
import base64
import csv
import io
import json
import os
import re
import sys
import requests


# === 讀取地標資料庫 ===

SHEET_CSV_URL = "https://docs.google.com/spreadsheets/d/15Xes5VuHMcg8r-mQesv829VnXU1JR1NUHCtyQfYZhCY/export?format=csv"


def load_landmarks(csv_url=SHEET_CSV_URL):
    """從 Google Sheet 讀取地標資料，去重後回傳"""
    resp = requests.get(csv_url)
    resp.encoding = "utf-8"
    reader = csv.DictReader(io.StringIO(resp.text))

    locations = {}
    for r in reader:
        name = r.get("地點名稱 (name)", "").strip()
        if not name or name in locations:
            continue
        locations[name] = {
            "name": name,
            "style": r.get("建築風格 (style)", ""),
            "struct": r.get("建築結構 (struct)", ""),
            "material": r.get("材質 (material)", ""),
            "function": r.get("功能用途 (function)", ""),
            "summary": r.get("簡介 (summary)", "")[:200],
        }

    return locations


def build_context(locations):
    """把地標資料組成 prompt context"""
    context = f"虎尾地標資料庫（共 {len(locations)} 個地點）：\n\n"
    for i, (name, info) in enumerate(locations.items(), 1):
        context += f"{i}. {name}\n"
        if info["style"]:
            context += f"   風格：{info['style']}\n"
        if info["struct"]:
            context += f"   結構：{info['struct']}\n"
        if info["material"]:
            context += f"   材質：{info['material']}\n"
        if info["function"]:
            context += f"   用途：{info['function']}\n"
        if info["summary"]:
            context += f"   簡介：{info['summary'][:120]}\n"
        context += "\n"
    return context


# === HackMD ===

def load_images_from_hackmd(hackmd_url):
    """從 HackMD 頁面抓取所有圖片 URL"""
    # 從 URL 取得 note ID
    # https://hackmd.io/@yillkid/Hy42AcLj-g → Hy42AcLj-g
    match = re.search(r'hackmd\.io/(?:@[^/]+/)?([a-zA-Z0-9_-]+)', hackmd_url)
    if not match:
        print(f"無法解析 HackMD URL：{hackmd_url}")
        return []

    note_id = match.group(1)

    # 嘗試用公開 API 讀取
    resp = requests.get(f"https://hackmd.io/{note_id}/download")
    if resp.status_code != 200:
        # fallback: 直接抓 HTML
        resp = requests.get(hackmd_url)

    content = resp.text

    # 找所有 HackMD 上傳的圖片
    urls = re.findall(r'https://hackmd\.io/_uploads/[a-zA-Z0-9_-]+\.(?:png|jpg|jpeg)', content)

    # 去重但保持順序
    seen = set()
    unique = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            unique.append(url)

    return unique


# === Gemini API ===

def detect_landmark(image_bytes, context, api_key):
    """用 Gemini 辨識圖片中的地標"""
    img_b64 = base64.b64encode(image_bytes).decode()

    prompt = f"""{context}

請根據照片的視覺特徵（建築風格、材質、文字、雕塑、場景），從上面的地點清單中選出最匹配的一個。

規則：
- 照片一定是上面的地點之一，不要回答「不確定」
- 如果照片上有文字（碑文、門牌、布條），優先用文字判斷
- 如果沒有文字，用建築風格、材質、結構來比對

請用以下 JSON 格式回答：
{{"name": "地點名稱", "reason": "判斷依據", "confidence": "high/medium/low"}}

只輸出 JSON。"""

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"

    resp = requests.post(url, json={
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/png", "data": img_b64}},
            ]
        }],
        "generationConfig": {
            "response_mime_type": "application/json",
        },
    })

    data = resp.json()

    if "error" in data:
        return {"error": data["error"]["message"]}

    text = data["candidates"][0]["content"]["parts"][0]["text"]
    return json.loads(text)


# === 圖片載入 ===

def load_image_from_file(path):
    """從本地檔案讀取圖片"""
    with open(path, "rb") as f:
        return f.read()


def load_image_from_url(url):
    """從 URL 下載圖片"""
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.content


# === 主程式 ===

def main():
    parser = argparse.ArgumentParser(description="RAG Vision — 地標辨識")
    parser.add_argument("images", nargs="*", help="圖片檔案、資料夾或 URL")
    parser.add_argument("--hackmd", help="從 HackMD 頁面批次載入圖片")
    parser.add_argument("--key", default=os.getenv("GEMINI_API_KEY"), help="Gemini API Key")
    args = parser.parse_args()

    if not args.key:
        print("錯誤：請設定 GEMINI_API_KEY 環境變數，或用 --key 指定")
        sys.exit(1)

    if not args.images and not args.hackmd:
        print("錯誤：請提供圖片或 --hackmd URL")
        parser.print_help()
        sys.exit(1)

    # 讀取地標資料庫
    print("讀取地標資料庫...")
    locations = load_landmarks()
    context = build_context(locations)
    print(f"載入 {len(locations)} 個地點\n")

    # 收集圖片
    tasks = []

    # 從 HackMD 頁面抓圖
    if args.hackmd:
        print(f"從 HackMD 載入圖片：{args.hackmd}")
        urls = load_images_from_hackmd(args.hackmd)
        print(f"找到 {len(urls)} 張圖片\n")
        for url in urls:
            tasks.append(("url", url))

    # 從參數收集
    for path in args.images:
        if path.startswith("http://") or path.startswith("https://"):
            tasks.append(("url", path))
        elif os.path.isdir(path):
            for f in sorted(os.listdir(path)):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    tasks.append(("file", os.path.join(path, f)))
        elif os.path.isfile(path):
            tasks.append(("file", path))
        else:
            print(f"找不到：{path}")

    if not tasks:
        print("沒有找到任何圖片")
        sys.exit(1)

    # 逐張辨識
    results = []

    for i, (source_type, source) in enumerate(tasks, 1):
        label = os.path.basename(source) if source_type == "file" else source.split("/")[-1]
        print(f"[{i}/{len(tasks)}] {label}")

        try:
            if source_type == "file":
                image_bytes = load_image_from_file(source)
            else:
                image_bytes = load_image_from_url(source)

            result = detect_landmark(image_bytes, context, args.key)

            if "error" in result:
                print(f"  錯誤：{result['error']}")
            else:
                print(f"  地點：{result['name']}")
                print(f"  依據：{result['reason']}")
                print(f"  信心：{result.get('confidence', 'N/A')}")
                results.append(result)

        except Exception as e:
            print(f"  錯誤：{e}")

        print()

    # 總結
    if results:
        print(f"{'=' * 50}")
        print(f"  辨識完成：{len(results)}/{len(tasks)} 張")
        print(f"{'=' * 50}")
        for i, r in enumerate(results, 1):
            print(f"  {i}. {r['name']} ({r.get('confidence', '?')})")


if __name__ == "__main__":
    main()
