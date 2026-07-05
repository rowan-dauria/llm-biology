# Hosting the Graph Viewer on Cloudflare Pages

Use Cloudflare Pages Direct Upload to host the static graph viewer. The hosted
copy is view-only; keep graph editing in the local working UI, save the edited
graph JSONs, then regenerate this static bundle.

## Build the Static Viewer

From the `llm-biology` repo:

```bash
cd /Users/rowandauria/Documents/GitHub/mphil-project/llm-biology

/Users/rowandauria/miniconda3/envs/qwen-sae-mac/bin/python -m llm_biology.viewer.export_static \
  --output-dir dist/graph-viewer \
  --graph-file-dir ../data/llm-biology/ui_graphs
```

This writes the deployable site to:

```text
/Users/rowandauria/Documents/GitHub/mphil-project/llm-biology/dist/graph-viewer
```

The exporter only copies feature sidecar JSON files referenced by the exported
graphs, so the bundle should stay below Cloudflare's Direct Upload file-count
limit. If you add more graphs, rerun the export and check the file count before
uploading:

```bash
find dist/graph-viewer -type f | wc -l
```

## Upload to Cloudflare Pages

1. Open the Cloudflare Dashboard.
2. Go to **Workers & Pages**.
3. Choose **Create application**.
4. Choose **Pages**.
5. Choose **Upload assets** / **Direct Upload**.
6. Use a project name such as `llm-biology-graphs`.
7. Upload the contents of `dist/graph-viewer`, with `index.html` at the site root.

Do not upload a parent folder that places `index.html` under
`graph-viewer/index.html`; Cloudflare should receive `index.html`,
`biology-server.js`, `biology-server.css`, `ct/`, `data/`, and `graph_data/`
at the root of the uploaded site.

## Sanity Check

After Cloudflare gives a Pages URL, check:

- The graph dropdown shows the expected slugs.
- Each graph loads without a blank screen.
- Top activating windows appear in the feature detail panel.
- The hosted copy does not need upload/save/edit controls.

## Updating the Hosted Viewer

After any final graph edits:

1. Save the graph in the local editing UI.
2. Rerun the static export command above.
3. Upload the regenerated `dist/graph-viewer` contents to Cloudflare Pages again.
