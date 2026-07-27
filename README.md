# Model Explorer

Cluster-aware Hugging Face model finder for **GGUF** and **MLX**.

I found that HF’s hardware picker assumes just a single machine. This tool maps multi-host RAM, memory bandwidth, and interconnect to models that fit, with theoretical decode tok/s, so you can find your models easily. 

**Live demo:** https://caseytaylorlearningjourney.github.io/HF_ModelExplorer/

## What you dial in

- **Hosts** — RAM, optional usable-GB override, mem bandwidth presets (Apple M1–M5 base/Pro/Max, Ultras where they exist, Strix Halo, RTX 3090/4090/5090, or Custom)
- **Quant** — effective-bits table (IQ4_XS, Q4_K_M, Q5_K_M, Q6_K, Q8_0, MXFP4, FP8, BF16, MLX 4–8 bit)
- **Context** — 4k through 1M · KV dtype FP16/FP8
- **Interconnect** — USB4/TB4 through 800GbE · shows compute vs network bound

Results: Fits / Tight / Won’t, plus est. tok/s (MoE uses active params). Snapshots rebuild every 2 hours on a fork with Actions; browser Refresh hits CDN only, never Hugging Face.

## How estimates work

Memory ≈ weights (effective bits) + KV(context) + overhead floor.  
Usable GB blank = auto (Apple ~75% Metal cap, others 1 − overhead%). Set it explicitly if you raised `iogpu.wired_limit_mb` or a GTT carve-ou for strix halo or GB10 models. 
tok/s ≈ memory bandwidth / bytes read per token. Cluster decode is a sequential pipeline (stage times sum, plus per-hop transfer + latency). Numbers are theoretical ceilings — details live in the on-page formula hint.

## Local build

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python ModelExplorer.py
python -m http.server 8765 --directory _site
# open http://127.0.0.1:8765
```

Optional: `export HF_TOKEN=hf_...` for authenticated crawls (higher rate limits).

Fork the repo if you want GitHub Pages + the every-2h snapshot rebuild; otherwise just build locally as above. No tokens are written into `index.html` or `data/*.json`.

## License

MIT
