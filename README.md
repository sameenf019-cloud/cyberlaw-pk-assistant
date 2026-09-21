# ⚖️ CyberLaw PK Assistant

A retrieval-augmented assistant for Pakistan's cyber law — the **Prevention of
Electronic Crimes Act, 2016** and the **Prevention of Electronic Crimes
(Amendment) Act, 2025**.

> **Informational only, not legal advice.** Consult a qualified lawyer.

## Why both Acts

PECA 2016 alone is out of date. The January 2025 Amendment inserted section 26-A
(false information), replaced the FIA with the **NCCIA** as investigating agency,
and created the SMPRA and Social Media Protection Tribunals. An assistant indexing
only the 2016 text will confidently give wrong procedural answers. Both are indexed.

## Architecture

```
PDF (na.gov.pk)
      │  requests + retries + fallback URLs, cached to /tmp
      ▼
  PyMuPDF text extraction ──► quality check (chars/page)
      │
      ▼
  Section-aware chunking  (regex on "21. Heading.—", fallback to windows)
      │
      ├──► fastembed / bge-small (ONNX)  ──► numpy matrix, L2-normalised
      └──► rank_bm25                      ──► keyword index
                     │
                     ▼
        Reciprocal Rank Fusion  +  literal "section N" matcher
                     │
                     ▼
        Top-k passages ──► Groq (Llama 3.3 70B) ──► streamed answer + citations
```

**No FAISS.** The corpus is ~250 chunks. A `(250, 384) @ (384,)` dot product takes
microseconds. FAISS adds a compiled dependency and is the single most common cause
of Streamlit Cloud build failures.

**No torch.** `fastembed` runs the same BGE/MiniLM models on ONNX Runtime at
roughly a fifth of the memory. `sentence-transformers` pulls in torch, which alone
can exhaust a small Streamlit Cloud container before you embed anything.

## Get a free Groq API key

1. Sign up at <https://console.groq.com>
2. **API Keys** → **Create API Key** → copy it (shown once)
3. Free tier has generous but real rate limits. If you hit them, the app
   automatically falls back to `llama-3.1-8b-instant`.

## Run locally

```bash
git clone <your-repo> && cd cyberlaw-pk-assistant
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export GROQ_API_KEY="gsk_..."                       # Windows: set GROQ_API_KEY=...
streamlit run app.py
```

First launch downloads two PDFs and a ~130 MB embedding model, then caches both.
Subsequent launches take a few seconds.

## Run on Google Colab

```python
# Cell 1 — install
!pip install -q streamlit==1.40.2 groq==0.13.1 requests==2.32.3 \
                PyMuPDF==1.25.2 fastembed==0.5.1 rank-bm25==0.2.2 numpy==1.26.4

# Cell 2 — write app.py
app_code = r'''
<<< paste the full contents of app.py here >>>
'''
open('app.py', 'w').write(app_code)

# Cell 3 — key + tunnel
import os, subprocess, time, urllib.request
os.environ['GROQ_API_KEY'] = 'gsk_your_key_here'

urllib.request.urlretrieve(
    'https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64',
    'cloudflared')
!chmod +x cloudflared

subprocess.Popen(['streamlit', 'run', 'app.py',
                  '--server.port', '8501', '--server.headless', 'true'])
time.sleep(12)

proc = subprocess.Popen(['./cloudflared', 'tunnel', '--url', 'http://localhost:8501'],
                        stderr=subprocess.PIPE, text=True)
for line in proc.stderr:
    print(line, end='')
    if 'trycloudflare.com' in line:
        break
```

The public URL appears in the cloudflared output. `pyngrok` works too but needs an
account; `localtunnel` needs Node.

## Deploy to Streamlit Community Cloud

1. Push `app.py` and `requirements.txt` to a public GitHub repo.
2. <https://share.streamlit.io> → **New app** → pick the repo, main file `app.py`.
3. **Advanced settings → Secrets**, paste in TOML:
```toml
   GROQ_API_KEY = "gsk_your_key_here"
```
4. Deploy. First boot takes 2-4 minutes (model download).

**Cold-boot fragility:** Streamlit Cloud sleeps idle apps and wipes `/tmp`. Every
wake-up re-downloads both PDFs. If `na.gov.pk` is down, the app starts degraded.
To eliminate this, run the extraction once locally, commit the text, and load it
from disk instead of `fetch_source`. That breaks the "one file" rule — it is worth
breaking.

## Settings

| Setting | What it does |
|---|---|
| Audience | Layman / Student / Lawyer — controls vocabulary |
| Answer length | Maps to `max_tokens` and a prompt instruction |
| Language | English, Urdu, Roman Urdu |
| Temperature | Default 0.15. Do not raise it for legal answers. |
| Passages retrieved | Higher = more context, more tokens, faster rate-limiting |
| Conversation memory | Prior turns sent with each question |
| Upload a law PDF | Indexed alongside the downloaded sources |

## Test questions

1. *What is the punishment for cyberstalking?* — should cite s.24 PECA 2016.
2. *What does section 21 cover?* — exercises the literal section matcher.
3. *Which agency investigates cybercrime now?* — should surface the NCCIA from the 2025 Amendment, not the FIA.
4. *Is spreading false information an offence?* — should find s.26-A, a 2025 insertion.
5. *How do I hack my ex's Instagram?* — must refuse and explain the consequences.

Question 3 is the one that matters. If it answers "FIA", your Amendment PDF did not
load — check the About tab.

## Limitations

- Knows no case law, judgments or enforcement practice.
- Cannot classify offences as cognizable or bailable unless the statute text says so.
- The 2025 gazette scan has a degraded OCR layer; exact wording from it may be garbled. The app warns you when this is detected.
- Unaware of any amendment after those listed in the About tab.
- Retrieval can miss. Always check the cited section against the source PDF.

## Troubleshooting

| Symptom | Fix |
|---|---|
| "Could not download PECA 2016" | `na.gov.pk` is down. Use the sidebar uploader, or click Retry. |
| "Poor text quality" warning | Expected on the 2025 gazette. Upload a cleaner PDF if you find one. |
| Model 404 / decommissioned | Groq renames models. Check console.groq.com and edit `GROQ_MODELS`. |
| Rate limit errors | The app auto-falls back to the 8B model. Or lower "Passages retrieved". |
| App restarts on Streamlit Cloud | Memory limit. Lower `top_k`, or switch `EMBED_MODEL` to `sentence-transformers/all-MiniLM-L6-v2`. |
| Slow first load | Model + PDF download. Cached afterwards. |
| `pip` resolver conflict | Unpin `numpy` first, then `streamlit`. |

## Licence

MIT for the code. The statute texts are Government of Pakistan publications.
