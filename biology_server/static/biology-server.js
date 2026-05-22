;(function () {
  const els = {
    prompt: document.querySelector('#prompt'),
    slug: document.querySelector('#slug'),
    maxFeatureNodes: document.querySelector('#max-feature-nodes'),
    edgeTopK: document.querySelector('#edge-top-k'),
    uploadFile: document.querySelector('#upload-file'),
    previewButton: document.querySelector('#preview-button'),
    generateButton: document.querySelector('#generate-button'),
    uploadButton: document.querySelector('#upload-button'),
    status: document.querySelector('#status'),
    targetToken: document.querySelector('#target-token'),
    topTokens: document.querySelector('#top-tokens'),
    jobLog: document.querySelector('#job-log'),
    graphFrame: document.querySelector('#graph-frame'),
  }

  let preview = null
  let activeJobId = null

  els.previewButton.addEventListener('click', previewPrompt)
  els.generateButton.addEventListener('click', generateGraph)
  els.uploadFile.addEventListener('change', () => {
    els.uploadButton.disabled = !selectedUploadFile()
  })
  els.uploadButton.addEventListener('click', uploadGraph)

  const initialSlug = getUrlParam('slug')
  if (initialSlug) renderGraph(initialSlug)

  async function previewPrompt() {
    setBusy(true)
    els.generateButton.disabled = true
    els.jobLog.textContent = ''
    setStatus('Previewing next token...')
    try {
      preview = await postJson('/api/preview', {
        prompt: els.prompt.value,
        slug: cleanValue(els.slug.value),
      })
      renderPreview(preview)
      els.generateButton.disabled = false
      setStatus('Preview ready')
    } catch (err) {
      preview = null
      renderError(err)
    } finally {
      setBusy(false)
    }
  }

  async function generateGraph() {
    if (!preview) return
    setBusy(true)
    els.generateButton.disabled = true
    setStatus('Queued graph generation...')
    try {
      const job = await postJson('/api/graphs', {
        preview_id: preview.preview_id,
        slug: cleanValue(els.slug.value),
        max_feature_nodes: Number(els.maxFeatureNodes.value),
        edge_top_k: Number(els.edgeTopK.value),
      })
      activeJobId = job.job_id
      pollJob(job.job_id)
    } catch (err) {
      setBusy(false)
      els.generateButton.disabled = false
      renderError(err)
    }
  }

  async function pollJob(jobId) {
    if (activeJobId !== jobId) return
    try {
      const job = await getJson(`/api/jobs/${jobId}`)
      renderJob(job)
      if (job.status === 'succeeded') {
        setBusy(false)
        els.generateButton.disabled = false
        renderGraph(job.slug)
        return
      }
      if (job.status === 'failed') {
        setBusy(false)
        els.generateButton.disabled = false
        setStatus('Graph generation failed')
        return
      }
      window.setTimeout(() => pollJob(jobId), 1500)
    } catch (err) {
      setBusy(false)
      els.generateButton.disabled = false
      renderError(err)
    }
  }

  async function uploadGraph() {
    const file = selectedUploadFile()
    if (!file) {
      setStatus('Choose a graph JSON file')
      return
    }

    setBusy(true)
    activeJobId = null
    preview = null
    els.generateButton.disabled = true
    els.jobLog.textContent = ''
    setStatus('Uploading graph...')

    try {
      let graph
      try {
        graph = JSON.parse(await file.text())
      } catch (_err) {
        throw new Error('Upload file must be valid JSON')
      }

      const uploaded = await postJson('/api/upload_graph', {
        graph,
        slug: cleanValue(els.slug.value),
        filename: file.name,
      })
      renderPreview(null)
      setStatus(`Uploaded: ${uploaded.slug}`)
      renderGraph(uploaded.slug)
    } catch (err) {
      renderError(err)
    } finally {
      setBusy(false)
    }
  }

  function renderPreview(data) {
    if (!data) {
      els.targetToken.textContent = ''
      els.topTokens.innerHTML = ''
      return
    }
    const token = data.target_token
    els.targetToken.textContent = `Target: ${JSON.stringify(token.text)}  id=${token.id}  p=${formatProb(token.prob)}`
    els.topTokens.innerHTML = ''
    data.top_tokens.forEach(item => {
      const chip = document.createElement('span')
      chip.className = 'token-chip'
      chip.textContent = `${JSON.stringify(item.text)} ${formatProb(item.prob)}`
      els.topTokens.appendChild(chip)
    })
  }

  function renderJob(job) {
    const bits = [`${job.status}: ${job.slug}`]
    if (job.feature_nodes !== null && job.feature_nodes !== undefined) {
      bits.push(`${job.feature_nodes} nodes`)
    }
    if (job.links !== null && job.links !== undefined) {
      bits.push(`${job.links} links`)
    }
    setStatus(bits.join(' · '))
    els.jobLog.textContent = (job.logs || []).slice(-18).join('\n')
    if (job.error) {
      els.jobLog.textContent = `${job.error}\n${els.jobLog.textContent}`
    }
  }

  function renderGraph(slug) {
    setUrlParam('slug', slug)
    els.graphFrame.src = `/ct/index.html?slug=${encodeURIComponent(slug)}`
    document.title = `Attribution Graph: ${slug}`
  }

  async function postJson(url, body) {
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
    const data = await res.json().catch(() => ({}))
    if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`)
    return data
  }

  async function getJson(url) {
    const res = await fetch(url)
    const data = await res.json().catch(() => ({}))
    if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`)
    return data
  }

  function cleanValue(value) {
    const trimmed = value.trim()
    return trimmed ? trimmed : undefined
  }

  function selectedUploadFile() {
    return els.uploadFile.files && els.uploadFile.files[0]
  }

  function setBusy(isBusy) {
    els.previewButton.disabled = isBusy
    els.uploadButton.disabled = isBusy || !selectedUploadFile()
  }

  function setStatus(text) {
    els.status.textContent = text
  }

  function renderError(err) {
    setStatus(err.message || String(err))
  }

  function getUrlParam(key) {
    return new URL(window.location.href).searchParams.get(key)
  }

  function setUrlParam(key, value) {
    const url = new URL(window.location.href)
    if (value === null || value === undefined || value === '') {
      url.searchParams.delete(key)
    } else {
      url.searchParams.set(key, value)
    }
    window.history.replaceState(null, '', url)
  }

  function formatProb(value) {
    return Number(value).toFixed(4)
  }
})()
