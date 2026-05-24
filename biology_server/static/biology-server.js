;(function () {
  const els = {
    prompt: document.querySelector('#prompt'),
    slug: document.querySelector('#slug'),
    maxFeatureNodes: document.querySelector('#max-feature-nodes'),
    edgeTopK: document.querySelector('#edge-top-k'),
    useChatTemplate: document.querySelector('#use-chat-template'),
    uploadFile: document.querySelector('#upload-file'),
    previewButton: document.querySelector('#preview-button'),
    generateButton: document.querySelector('#generate-button'),
    uploadButton: document.querySelector('#upload-button'),
    status: document.querySelector('#status'),
    targetToken: document.querySelector('#target-token'),
    topTokens: document.querySelector('#top-tokens'),
    jobLog: document.querySelector('#job-log'),
  }

  let preview = null
  let activeJobId = null
  const graphStateParamKeys = [
    'pinnedIds',
    'supernodes',
    'linkType',
    'clickedId',
    'sg_pos',
    'pruningThreshold',
    'clerps',
  ]

  els.previewButton.addEventListener('click', previewPrompt)
  els.generateButton.addEventListener('click', generateGraph)
  els.useChatTemplate.addEventListener('change', updateGenerateButton)
  els.uploadFile.addEventListener('change', () => {
    els.uploadButton.disabled = !selectedUploadFile()
  })
  els.uploadButton.addEventListener('click', uploadGraph)

  const graphView = createGraphView(d3.select('.nav'), d3.select('#graph'))

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
        use_chat_template: els.useChatTemplate.checked,
      })
      renderPreview(preview)
      updateGenerateButton()
      setStatus('Preview ready')
    } catch (err) {
      preview = null
      renderError(err)
    } finally {
      setBusy(false)
    }
  }

  async function generateGraph() {
    if (!preview || preview.use_chat_template !== els.useChatTemplate.checked) return
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

  function updateGenerateButton() {
    els.generateButton.disabled =
      !preview || preview.use_chat_template !== els.useChatTemplate.checked
  }

  async function pollJob(jobId) {
    if (activeJobId !== jobId) return
    try {
      const job = await getJson(`/api/jobs/${jobId}`)
      renderJob(job)
      if (job.status === 'succeeded') {
        setBusy(false)
        els.generateButton.disabled = false
        renderGraph(job.slug, { resetGraphState: true })
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
      renderGraph(uploaded.slug, { resetGraphState: true })
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

  function renderGraph(slug, { resetGraphState = false } = {}) {
    if (resetGraphState) clearGraphStateParams()
    util.params.set('slug', slug)
    graphView.show(slug)
  }

  function clearGraphStateParams() {
    graphStateParamKeys.forEach(key => util.params.set(key, null))
  }

  // Ported from circuit-tracer's frontend index.html; keep in sync with it.
  function createGraphView(navSel, cgSel) {
    window.isLocalServing = true

    let graphs = []
    let visState = null
    let activeGraphData = null
    let sliderContainer = null
    let navBuilt = false
    const debouncedRender = util.debounce(render, 300)

    async function show(slug) {
      const meta = await util.getFile('./data/graph-metadata.json', false)
      const graphData = await util.getFile(`./graph_data/${slug}.json`, false)
      activeGraphData = graphData
      graphs = meta.graphs || []

      const currentGraph = graphs.find(g => g.slug === slug)
      seedGraphStateParams(graphData, currentGraph)

      visState = {
        slug,
        clickedId: util.params.get('clickedId')?.replace('null', ''),
        isGridsnap: util.params.get('isGridsnap')?.replace('null', ''),
      }

      if (currentGraph && typeof currentGraph.node_threshold === 'number') {
        const urlPruningThreshold = normalizePruningThreshold(
          util.params.get('pruningThreshold'),
          currentGraph.node_threshold
        )
        visState.pruningThreshold =
          urlPruningThreshold ||
          currentGraph.node_threshold ||
          0.4
      }

      buildNav()
      render()
    }

    function buildNav() {
      if (navBuilt) return
      navBuilt = true

      navSel
        .html('')
        .style('display', 'flex')
        .style('align-items', 'center')
        .style('justify-content', 'space-between')

      navSel
        .append('div.controls-container')
        .style('display', 'flex')
        .style('align-items', 'center')
        .style('flex', '1')
        .style('gap', '20px')

      navSel
        .append('button.save-button')
        .text('Save')
        .on('click', saveGraph)
    }

    function saveGraph() {
      const slug = visState && visState.slug
      if (!slug) {
        console.error('No slug found')
        return
      }

      const saveButton = navSel.select('.save-button')
      saveButton.text('Saving...').attr('disabled', true).style('opacity', 0.6)

      fetch(`/save_graph/${slug}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          qParams: Object.fromEntries(
            graphStateParamKeys
              .map(k => [k, util.params.get(k)])
              .filter(([_k, v]) => v !== undefined && v !== null && v !== 'null')
          ),
        }),
      })
        .then(response => {
          if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`)
          saveButton
            .text('Saved!')
            .style('background-color', '#e6f7e6')
            .style('border-color', '#8bc34a')
          setTimeout(resetSaveButton, 2000)
        })
        .catch(error => {
          console.error('Error saving graph:', error)
          saveButton
            .text('Error!')
            .style('background-color', '#ffebee')
            .style('border-color', '#f44336')
          setTimeout(resetSaveButton, 2000)
        })

      function resetSaveButton() {
        saveButton
          .text('Save')
          .attr('disabled', null)
          .style('opacity', null)
          .style('background-color', null)
          .style('border-color', null)
      }
    }

    function render() {
      const m = graphs.find(g => g.slug == visState.slug)
      if (!m) return

      const controlsContainer = navSel.select('.controls-container')

      if (typeof m.node_threshold === 'number') {
        if (!sliderContainer) {
          sliderContainer = controlsContainer
            .append('div.slider-container')
            .style('display', 'flex')
            .style('align-items', 'center')
            .style('gap', '8px')

          sliderContainer.append('span').text('Pruning:')

          sliderContainer
            .append('input')
            .attr('type', 'range')
            .attr('min', 0)
            .attr('max', m.node_threshold)
            .attr('step', 0.01)
            .attr('value', visState.pruningThreshold || m.node_threshold)
            .on('input', function () {
              visState.pruningThreshold = this.value
              visState.clickedId = util.params.get('clickedId')?.replace('null', '')
              util.params.set('pruningThreshold', this.value)
              sliderContainer
                .select('.value-display')
                .text(parseFloat(this.value).toFixed(2))
              debouncedRender()
            })

          sliderContainer.append('span.value-display')
        }

        sliderContainer
          .select('input')
          .attr('max', m.node_threshold)
          .property('value', visState.pruningThreshold || m.node_threshold)
        sliderContainer
          .select('.value-display')
          .text(parseFloat(visState.pruningThreshold || m.node_threshold).toFixed(2))
        sliderContainer.style('display', 'flex')

        if (visState.pruningThreshold === undefined) {
          visState.pruningThreshold = m.node_threshold
        }
      } else {
        if (sliderContainer) {
          sliderContainer.remove()
          sliderContainer = null
        }
        delete visState.pruningThreshold
        util.params.set('pruningThreshold', null)
      }

      syncActiveGraphPruning()
      cgSel.html('')

      const options = {
        clickedId: visState.clickedId,
        clickedIdCb: id => util.params.set('clickedId', id),
        isGridsnap: visState.isGridsnap || true,
      }
      if (visState.pruningThreshold !== undefined) {
        options.pruningThreshold = visState.pruningThreshold
      }

      initCg(cgSel, visState.slug, options)
      document.title = 'Attribution Graph: ' + m.prompt
    }

    function syncActiveGraphPruning() {
      if (!activeGraphData || typeof activeGraphData.qParams !== 'object') return
      if (visState.pruningThreshold === undefined) {
        delete activeGraphData.qParams.pruningThreshold
      } else {
        activeGraphData.qParams.pruningThreshold = String(visState.pruningThreshold)
      }
    }

    return { show }
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

  function normalizePruningThreshold(value, max) {
    if (value === undefined || value === null || value === '') return null
    const parsed = Number(value)
    if (!Number.isFinite(parsed)) return null
    const numericMax = Number(max)
    const upper = Number.isFinite(numericMax) ? numericMax : parsed
    return String(Math.min(Math.max(parsed, 0), upper))
  }

  function seedGraphStateParams(graphData, currentGraph) {
    const qParams = graphData && graphData.qParams
    if (!qParams || typeof qParams !== 'object') return

    graphStateParamKeys.forEach(key => {
      if (util.params.get(key)) return
      const value = graphStateParamToUrl(key, qParams[key], currentGraph)
      if (value !== null) util.params.set(key, value)
    })
  }

  function graphStateParamToUrl(key, value, currentGraph) {
    if (value === undefined || value === null) return null

    if (key === 'pruningThreshold') {
      return normalizePruningThreshold(value, currentGraph?.node_threshold)
    }

    if (key === 'pinnedIds') {
      if (Array.isArray(value)) {
        const ids = value.filter(item => typeof item === 'string' && item)
        return ids.length ? ids.join(',') : null
      }
      return typeof value === 'string' && value ? value : null
    }

    if (key === 'supernodes' || key === 'clerps') {
      if (Array.isArray(value)) return value.length ? JSON.stringify(value) : null
      return typeof value === 'string' && value ? value : null
    }

    return typeof value === 'string' && value ? value : null
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

  function formatProb(value) {
    return Number(value).toFixed(4)
  }
})()
