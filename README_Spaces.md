# Deploy as a public Gradio Space (free)

Get a **permanent public link** (e.g. `https://huggingface.co/spaces/YOUR_USER/combinatorial-opt-agent`) by deploying to Hugging Face Spaces.

## Option A: Gradio CLI (easiest)

1. Install: `pip install gradio`
2. Log in: `huggingface-cli login` (create a free account at [huggingface.co](https://huggingface.co) if needed)
3. From this repo root run:
   ```bash
   gradio deploy
   ```
4. Follow the prompts (Space name, SDK = Gradio). Your app will be at:
   **`https://huggingface.co/spaces/YOUR_USERNAME/SPACE_NAME`**

## Option B: Create Space in the browser

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Pick a name, choose **Gradio** as SDK, create the Space
3. Clone the Space repo, copy into it:
   - `app.py`
   - `requirements.txt`
   - `retrieval/` (folder)
   - `data/processed/all_problems.json`
   - `schema/` (if needed)
4. Push to the Space repo. The Space will build and give you a public URL.

## Option C: Run locally and get a temporary public link

From this repo (after `bash setup_catalog.sh`):

```bash
python app.py
```

The app is already set to `share=True`. In the terminal you’ll see something like:

```
Running on local URL:  http://127.0.0.1:7860
Running on public URL: https://XXXX.gradio.live
```

Open the **public URL** in any browser (or share it); it stays up while the app is running.
