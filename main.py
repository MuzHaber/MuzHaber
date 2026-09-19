"""
Muzhaber: Habertürk, CNN Türk ve TRT Haber'in aynı haberlerini
Gemini ile karşılaştırıp özetleyen günlük haber sitesi üreticisi.

Çalıştırma:  python main.py
Gerekli:     GEMINI_API_KEY ortam değişkeni
"""

import html
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import feedparser
from google import genai

# ---------------- AYARLAR ----------------
# Bir RSS adresi çalışmazsa sadece buradaki linki değiştirmeniz yeterli.
FEEDS = {
    "Habertürk": "https://www.haberturk.com/rss",
    "CNN Türk": "https://www.cnnturk.com/feed/rss/all/news",
    "TRT Haber": "https://www.trthaber.com/sondakika.rss",
}
ITEMS_PER_FEED = 25          # Her siteden kaç haber okunsun
MAX_STORIES = 5              # Günde en fazla kaç ortak haber işlensin
MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
OUT_DIR = Path("docs")
# ------------------------------------------


def clean(text: str) -> str:
    """HTML etiketlerini ve fazla boşlukları temizler."""
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def fetch_news() -> list[dict]:
    """Üç sitenin RSS akışını okur, numaralı bir liste döner."""
    items = []
    for source, url in FEEDS.items():
        try:
            feed = feedparser.parse(url)
            entries = feed.entries[:ITEMS_PER_FEED]
            if not entries:
                print(f"UYARI: {source} boş döndü ({url})")
            for e in entries:
                items.append(
                    {
                        "id": len(items),
                        "kaynak": source,
                        "baslik": clean(e.get("title", "")),
                        "aciklama": clean(e.get("summary", ""))[:300],
                        "link": e.get("link", ""),
                    }
                )
        except Exception as ex:  # bir site bozuksa diğerleri devam etsin
            print(f"UYARI: {source} okunamadı: {ex}")
    return items


def analyze(items: list[dict]) -> list[dict]:
    """Gemini'den ortak haberleri bulmasını ve özetlemesini ister."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit("HATA: GEMINI_API_KEY tanımlı değil.")

    listing = "\n".join(
        f"[{i['id']}] ({i['kaynak']}) {i['baslik']} - {i['aciklama']}" for i in items
    )
    prompt = f"""Aşağıda üç Türk haber sitesinden (Habertürk, CNN Türk, TRT Haber)
bugünün haber başlıkları var. Her satırın başındaki köşeli parantez içindeki sayı haberin numarasıdır.

Görevin:
1. AYNI olayı anlatan haberleri grupla. En az 2 farklı siteden gelen olayları seç,
   3 sitede de geçenleri öncelikle.
2. En önemli en fazla {MAX_STORIES} olayı seç.
3. Her olay için Türkçe yaz:
   - baslik: kısa, tarafsız bir başlık
   - ozet: 3-4 cümlelik, kendi cümlelerinle özet (kaynaklardan kopyalama yapma)
   - farklar: sitelerin vurgu veya bilgi farkları varsa 1-2 cümle, yoksa "Belirgin fark yok."
   - kaynak_numaralari: ilgili haberlerin numaraları (sayı listesi)
Sadece verilen bilgilere dayan, bilgi uydurma.

Yanıtı SADECE şu JSON formatında ver:
{{"haberler": [{{"baslik": "", "ozet": "", "farklar": "", "kaynak_numaralari": [0, 1]}}]}}

HABERLER:
{listing}
"""
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config={"response_mime_type": "application/json"},
    )
    text = re.sub(r"^```(?:json)?|```$", "", response.text.strip(), flags=re.M).strip()
    data = json.loads(text)

    by_id = {i["id"]: i for i in items}
    stories = []
    for s in data.get("haberler", []):
        sources = [by_id[n] for n in s.get("kaynak_numaralari", []) if n in by_id]
        if len(sources) < 2:
            continue
        s["kaynaklar"] = sources
        stories.append(s)
    return stories


STYLE = """
body{font-family:system-ui,Segoe UI,Arial,sans-serif;max-width:760px;margin:0 auto;padding:20px;
line-height:1.6;color:#1a1a1a;background:#fafafa}
h1{margin-bottom:0}.sub{color:#666;margin-top:4px}
article{background:#fff;border:1px solid #e3e3e3;border-radius:10px;padding:16px 20px;margin:18px 0}
h2{margin-top:0;font-size:1.2rem}.fark{color:#555;font-size:.95rem}
.src a{display:inline-block;margin-right:12px;font-size:.9rem}
a{color:#0a58ca}footer{color:#888;font-size:.85rem;margin-top:30px}
"""


def page(title: str, body: str) -> str:
    return (
        "<!doctype html><html lang='tr'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{html.escape(title)}</title><style>{STYLE}</style></head>"
        f"<body>{body}</body></html>"
    )


def render_day(date_str: str, stories: list[dict]) -> str:
    parts = [f"<h1>Muzhaber</h1><p class='sub'>{date_str} günlük haber özeti</p>"]
    for s in stories:
        links = "".join(
            f"<a href='{html.escape(x['link'])}' target='_blank' rel='noopener'>"
            f"{html.escape(x['kaynak'])}</a>"
            for x in s["kaynaklar"]
        )
        parts.append(
            f"<article><h2>{html.escape(s['baslik'])}</h2>"
            f"<p>{html.escape(s['ozet'])}</p>"
            f"<p class='fark'><b>Kaynaklar arası fark:</b> {html.escape(s['farklar'])}</p>"
            f"<p class='src'>{links}</p></article>"
        )
    parts.append(
        "<footer>Özetler yapay zeka (Gemini) tarafından üretilmiştir, hata içerebilir. "
        "Ayrıntı için kaynak haberlere bakın. <a href='index.html'>Tüm günler</a></footer>"
    )
    return page(f"Muzhaber {date_str}", "".join(parts))


def render_index() -> str:
    days = sorted(
        (p.stem for p in OUT_DIR.glob("20??-??-??.html")), reverse=True
    )
    lis = "".join(f"<li><a href='{d}.html'>{d}</a></li>" for d in days)
    body = (
        "<h1>Muzhaber</h1><p class='sub'>Habertürk, CNN Türk ve TRT Haber'in "
        "ortak haberleri, yapay zeka özetiyle.</p>"
        f"<h2>Günler</h2><ul>{lis}</ul>"
    )
    return page("Muzhaber", body)


def main() -> None:
    today = datetime.now(ZoneInfo("Europe/Istanbul")).strftime("%Y-%m-%d")
    items = fetch_news()
    print(f"{len(items)} haber okundu.")
    if not items:
        sys.exit("HATA: hiç haber okunamadı.")

    stories = analyze(items)
    print(f"{len(stories)} ortak haber bulundu.")
    if not stories:
        sys.exit("HATA: ortak haber bulunamadı.")

    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / f"{today}.html").write_text(render_day(today, stories), encoding="utf-8")
    (OUT_DIR / "index.html").write_text(render_index(), encoding="utf-8")
    print("Tamam.")


if __name__ == "__main__":
    main()
