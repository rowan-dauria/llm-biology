;(function () {
  const DEFAULT_INFLUENCE_PERCENT = 60

  const els = {
    graphSelect: document.querySelector('#graph-select'),
    loadButton: document.querySelector('#load-button'),
    promptDrawer: document.querySelector('#prompt-drawer'),
    prompt: document.querySelector('#prompt'),
    slug: document.querySelector('#slug'),
    maxFeatureNodes: document.querySelector('#max-feature-nodes'),
    edgeTopK: document.querySelector('#edge-top-k'),
    uploadFile: document.querySelector('#upload-file'),
    previewButton: document.querySelector('#preview-button'),
    generateButton: document.querySelector('#generate-button'),
    uploadButton: document.querySelector('#upload-button'),
    saveButton: document.querySelector('#save-button'),
    influenceCutoff: document.querySelector('#influence-cutoff'),
    influenceLabel: document.querySelector('#influence-label'),
    status: document.querySelector('#status'),
    targetToken: document.querySelector('#target-token'),
    topTokens: document.querySelector('#top-tokens'),
    jobLog: document.querySelector('#job-log'),
    graph: document.querySelector('#graph'),
  }

  const state = {
    graphs: [],
    preview: null,
    activeJobId: null,
    currentSlug: null,
    renderSeq: 0,
    influenceOverride: Boolean(util.params.get('influenceCutoff')),
    influenceTimer: null,
  }

  els.graphSelect.addEventListener('change', () => renderSelectedGraph())
  els.graphSelect.addEventListener('click', () => loadGraphList({ preserveSelection: true }))
  els.loadButton.addEventListener('click', () => renderSelectedGraph({ forceMetadata: true }))
  els.previewButton.addEventListener('click', previewPrompt)
  els.generateButton.addEventListener('click', generateGraph)
  els.uploadFile.addEventListener('change', () => {
    els.uploadButton.disabled = !selectedUploadFile()
  })
  els.uploadButton.addEventListener('click', uploadGraph)
  els.saveButton.addEventListener('click', saveGraph)
  els.influenceCutoff.addEventListener('input', () => {
    state.influenceOverride = true
    renderInfluenceLabel()
    util.params.set('influenceCutoff', currentInfluenceCutoff())
    window.clearTimeout(state.influenceTimer)
    state.influenceTimer = window.setTimeout(() => {
      if (state.currentSlug) renderGraph(state.currentSlug)
    }, 250)
  })

  init()

  async function init() {
    setInfluenceControl(util.params.get('influenceCutoff') || DEFAULT_INFLUENCE_PERCENT/100)
    await loadGraphList()
    const initialSlug = util.params.get('slug') || state.graphs[0]?.slug
    if (initialSlug) {
      ensureGraphOption(initialSlug)
      els.graphSelect.value = initialSlug
      renderGraph(initialSlug)
    } else {
      setStatus('No graphs found')
    }
  }

  async function loadGraphList({ preserveSelection = false } = {}) {
    const previous = preserveSelection ? els.graphSelect.value : null
    try {
      const metadata = await getJson('/data/graph-metadata.json')
      state.graphs = Array.isArray(metadata.graphs) ? metadata.graphs : []
      renderGraphOptions(previous)
    } catch (err) {
      renderError(err)
    }
  }

  function renderGraphOptions(preferredSlug) {
    const current = preferredSlug || state.currentSlug || util.params.get('slug')
    els.graphSelect.innerHTML = ''
    state.graphs.forEach(graph => {
      const option = document.createElement('option')
      option.value = graph.slug
      option.textContent = graphLabel(graph)
      option.title = graph.prompt || graph.slug
      els.graphSelect.appendChild(option)
    })
    if (current) ensureGraphOption(current)
    if (current && [...els.graphSelect.options].some(option => option.value === current)) {
      els.graphSelect.value = current
    }
  }

  function ensureGraphOption(slug) {
    if (!slug || [...els.graphSelect.options].some(option => option.value === slug)) return
    const option = document.createElement('option')
    option.value = slug
    option.textContent = slug
    option.title = slug
    els.graphSelect.appendChild(option)
  }

  function graphLabel(graph) {
    const prompt = graph.prompt || graph.slug
    const scan = graph.scan ? `${graph.scan} - ` : ''
    return `${scan}${prompt}`
  }

  function renderSelectedGraph({ forceMetadata = false } = {}) {
    const slug = els.graphSelect.value
    if (!slug) return
    if (forceMetadata) loadGraphList({ preserveSelection: true })
    renderGraph(slug)
  }

  async function previewPrompt() {
    setBusy(true)
    els.generateButton.disabled = true
    els.jobLog.textContent = ''
    setStatus('Previewing next token...')
    try {
      state.preview = await postJson('/api/preview', {
        prompt: els.prompt.value,
        slug: cleanValue(els.slug.value),
      })
      renderPreview(state.preview)
      els.generateButton.disabled = false
      setStatus('Preview ready')
    } catch (err) {
      state.preview = null
      renderError(err)
    } finally {
      setBusy(false)
    }
  }

  async function generateGraph() {
    if (!state.preview) return
    setBusy(true)
    els.generateButton.disabled = true
    setStatus('Queued graph generation...')
    try {
      const job = await postJson('/api/graphs', {
        preview_id: state.preview.preview_id,
        slug: cleanValue(els.slug.value),
        max_feature_nodes: Number(els.maxFeatureNodes.value),
        edge_top_k: Number(els.edgeTopK.value),
      })
      state.activeJobId = job.job_id
      pollJob(job.job_id)
    } catch (err) {
      setBusy(false)
      els.generateButton.disabled = false
      renderError(err)
    }
  }

  async function pollJob(jobId) {
    if (state.activeJobId !== jobId) return
    try {
      const job = await getJson(`/api/jobs/${jobId}`)
      renderJob(job)
      if (job.status === 'succeeded') {
        setBusy(false)
        els.generateButton.disabled = false
        await loadGraphList()
        ensureGraphOption(job.slug)
        els.graphSelect.value = job.slug
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
    state.activeJobId = null
    state.preview = null
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
      await loadGraphList()
      ensureGraphOption(uploaded.slug)
      els.graphSelect.value = uploaded.slug
      setStatus(`Uploaded ${uploaded.slug}`)
      renderGraph(uploaded.slug)
    } catch (err) {
      renderError(err)
    } finally {
      setBusy(false)
    }
  }

  async function saveGraph() {
    if (!state.currentSlug) {
      setStatus('No graph loaded')
      return
    }

    setSaveState('Saving...', { disabled: true })
    try {
      await postJson(`/save_graph/${encodeURIComponent(state.currentSlug)}`, {
        qParams: currentQParams(),
        timestamp: new Date().toISOString(),
      })
      setSaveState('Saved', { className: 'save-ok' })
      window.setTimeout(() => setSaveState('Save'), 2000)
    } catch (err) {
      setSaveState('Error', { className: 'save-error' })
      renderError(err)
      window.setTimeout(() => setSaveState('Save'), 2000)
    }
  }

  async function renderGraph(slug) {
    const renderId = ++state.renderSeq
    state.currentSlug = slug
    ensureGraphOption(slug)
    els.graphSelect.value = slug
    setSaveState('Save')
    setStatus(`Loading ${slug}...`)
    window.__datacache = {}
    util.params.set('slug', slug)

    try {
      if (!state.influenceOverride) {
        const savedCutoff = await readSavedInfluenceCutoff(slug)
        setInfluenceControl(savedCutoff ?? DEFAULT_INFLUENCE_PERCENT/100)
      }
      if (renderId !== state.renderSeq) return

      util.params.set('influenceCutoff', currentInfluenceCutoff())
      els.graph.innerHTML = '<div class="graph-loading">Loading graph...</div>'
      await initCg(d3.select(els.graph), slug, {
        clickedId: cleanValue(util.params.get('clickedId') || ''),
        clickedIdCb: id => {
          if (id) util.params.set('clickedId', id)
          else util.params.set('clickedId', null)
        },
        isGridsnap: true,
        gridPreset: 'biology-workbench',
        showFeatureLogits: false,
        showActivationHistogram: false,
        influenceCutoff: currentInfluenceCutoff(),
      })
      if (renderId !== state.renderSeq) return
      syncInfluenceControlFromGraph()
      setStatus(`Loaded ${slug}`)
      document.title = `Biology Graph: ${slug}`
    } catch (err) {
      if (renderId !== state.renderSeq) return
      state.currentSlug = null
      setSaveState('Save', { disabled: true })
      renderError(err)
    }
  }

  async function readSavedInfluenceCutoff(slug) {
    try {
      const graph = await getJson(`/graph_data/${encodeURIComponent(slug)}.json`)
      return parseCutoff(graph.qParams?.influenceCutoff)
    } catch (_err) {
      return null
    }
  }

  function syncInfluenceControlFromGraph() {
    const cutoff = parseCutoff(window.__lastCgVisState?.influenceCutoff)
    if (cutoff !== null) setInfluenceControl(cutoff)
  }

  function currentQParams() {
    const graphState = window.__lastCgVisState || {}
    const qParams = {}

    if (Array.isArray(graphState.pinnedIds)) qParams.pinnedIds = graphState.pinnedIds
    if (Array.isArray(graphState.hiddenIds) && graphState.hiddenIds.length) {
      qParams.hiddenIds = graphState.hiddenIds
    }

    const supernodes = graphState.subgraph?.supernodes || graphState.supernodes
    if (Array.isArray(supernodes)) qParams.supernodes = supernodes

    ;['linkType', 'clickedId', 'sg_pos', 'pruningThreshold', 'influenceCutoff'].forEach(key => {
      const value = key == 'influenceCutoff'
        ? currentInfluenceCutoff()
        : graphState[key] ?? util.params.get(key)
      if (value !== undefined && value !== null && value !== 'null') qParams[key] = value
    })

    if (graphState.og_sg_pos && !qParams.sg_pos) qParams.sg_pos = graphState.og_sg_pos
    if (graphState.clerps instanceof Map) {
      qParams.clerps = JSON.stringify([...graphState.clerps])
    } else {
      const clerps = util.params.get('clerps')
      if (clerps) qParams.clerps = clerps
    }

    return qParams
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
    setStatus(bits.join(' - '))
    els.jobLog.textContent = (job.logs || []).slice(-18).join('\n')
    if (job.error) {
      els.jobLog.textContent = `${job.error}\n${els.jobLog.textContent}`
    }
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

  function selectedUploadFile() {
    return els.uploadFile.files && els.uploadFile.files[0]
  }

  function setBusy(isBusy) {
    els.previewButton.disabled = isBusy
    els.uploadButton.disabled = isBusy || !selectedUploadFile()
    els.saveButton.disabled = isBusy || !state.currentSlug
    els.loadButton.disabled = isBusy
  }

  function setStatus(text) {
    els.status.textContent = text
  }

  function setSaveState(text, { disabled = !state.currentSlug, className = '' } = {}) {
    els.saveButton.textContent = text
    els.saveButton.disabled = disabled
    els.saveButton.classList.toggle('save-ok', className === 'save-ok')
    els.saveButton.classList.toggle('save-error', className === 'save-error')
  }

  function setInfluenceControl(value) {
    const cutoff = parseCutoff(value) ?? DEFAULT_INFLUENCE_PERCENT/100
    els.influenceCutoff.value = String(Math.round(cutoff*100))
    renderInfluenceLabel()
  }

  function renderInfluenceLabel() {
    els.influenceLabel.textContent = `${els.influenceCutoff.value}%`
  }

  function currentInfluenceCutoff() {
    return Number(els.influenceCutoff.value)/100
  }

  function parseCutoff(value) {
    if (value === undefined || value === null || value === '' || value === 'null') return null
    let parsed = Number(value)
    if (!Number.isFinite(parsed)) return null
    if (parsed > 1) parsed = parsed/100
    return Math.max(0, Math.min(1, parsed))
  }

  function cleanValue(value) {
    const trimmed = value.trim()
    return trimmed ? trimmed : undefined
  }

  function renderError(err) {
    setStatus(err.message || String(err))
  }

  function formatProb(value) {
    return Number(value).toFixed(4)
  }
})()
