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
    saveButton: document.querySelector('#save-button'),
    status: document.querySelector('#status'),
    targetToken: document.querySelector('#target-token'),
    topTokens: document.querySelector('#top-tokens'),
    jobLog: document.querySelector('#job-log'),
    graph: document.querySelector('#graph'),
  }

  let preview = null
  let activeJobId = null
  let currentSlug = null

  els.previewButton.addEventListener('click', previewPrompt)
  els.generateButton.addEventListener('click', generateGraph)
  els.uploadFile.addEventListener('change', () => {
    els.uploadButton.disabled = !selectedUploadFile()
  })
  els.uploadButton.addEventListener('click', uploadGraph)
  els.saveButton.addEventListener('click', saveGraph)

  const initialSlug = util.params.get('slug')
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

  async function saveGraph() {
    if (!currentSlug) {
      setStatus('No graph loaded')
      return
    }

    setSaveState('Saving...', { disabled: true })
    try {
      await postJson(`/save_graph/${encodeURIComponent(currentSlug)}`, {
        qParams: currentQParams(),
        timestamp: new Date().toISOString(),
      })
      setSaveState('Saved!', { className: 'save-ok' })
      window.setTimeout(() => setSaveState('Save'), 2000)
    } catch (err) {
      setSaveState('Error!', { className: 'save-error' })
      renderError(err)
      window.setTimeout(() => setSaveState('Save'), 2000)
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
    currentSlug = slug
    els.saveButton.disabled = false
    setSaveState('Save')
    window.__datacache = {}
    util.params.set('slug', slug)
    els.graph.innerHTML = ''
    const graphPromise = initCg(d3.select(els.graph), slug, {
      clickedIdCb: id => util.params.set('clickedId', id),
      isGridsnap: true,
    })
    graphPromise.catch(err => {
      currentSlug = null
      setSaveState('Save', { disabled: true })
      renderError(err)
    })
    document.title = `Attribution Graph: ${slug}`
  }

  function currentQParams() {
    const state = window.__lastCgVisState || {}
    const qParams = {}

    if (Array.isArray(state.pinnedIds)) qParams.pinnedIds = state.pinnedIds
    if (Array.isArray(state.hiddenIds) && state.hiddenIds.length) {
      qParams.hiddenIds = state.hiddenIds
    }

    const supernodes = state.subgraph?.supernodes || state.supernodes
    if (Array.isArray(supernodes)) qParams.supernodes = supernodes

    ;['linkType', 'clickedId', 'sg_pos', 'pruningThreshold'].forEach(key => {
      const value = state[key] ?? util.params.get(key)
      if (value !== undefined && value !== null && value !== 'null') qParams[key] = value
    })

    if (state.og_sg_pos && !qParams.sg_pos) qParams.sg_pos = state.og_sg_pos
    if (state.clerps instanceof Map) {
      qParams.clerps = JSON.stringify([...state.clerps])
    } else {
      const clerps = util.params.get('clerps')
      if (clerps) qParams.clerps = clerps
    }

    return qParams
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
    els.saveButton.disabled = isBusy || !currentSlug
  }

  function setStatus(text) {
    els.status.textContent = text
  }

  function setSaveState(text, { disabled = !currentSlug, className = '' } = {}) {
    els.saveButton.textContent = text
    els.saveButton.disabled = disabled
    els.saveButton.classList.toggle('save-ok', className === 'save-ok')
    els.saveButton.classList.toggle('save-error', className === 'save-error')
  }

  function renderError(err) {
    setStatus(err.message || String(err))
  }

  function formatProb(value) {
    return Number(value).toFixed(4)
  }
})()
