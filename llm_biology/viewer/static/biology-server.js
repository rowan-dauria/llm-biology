;(function () {
  const els = {
    graphSelect: document.querySelector('#graph-select'),
    slug: document.querySelector('#slug'),
    uploadFile: document.querySelector('#upload-file'),
    uploadButton: document.querySelector('#upload-button'),
    status: document.querySelector('#status'),
  }

  const graphStateParamKeys = [
    'pinnedIds',
    'supernodes',
    'linkType',
    'clickedId',
    'sg_pos',
    'pruningThreshold',
    'clerps',
  ]

  const graphView = createGraphView(d3.select('.nav'), d3.select('#graph'))

  els.graphSelect.addEventListener('change', () => {
    const slug = els.graphSelect.value
    if (slug) renderGraph(slug, { resetGraphState: true })
  })
  els.uploadFile.addEventListener('change', () => {
    els.uploadButton.disabled = !selectedUploadFile()
  })
  els.uploadButton.addEventListener('click', uploadGraph)

  loadGraphList()

  async function loadGraphList(preferredSlug) {
    try {
      const meta = await getJson('/data/graph-metadata.json')
      const graphs = Array.isArray(meta.graphs) ? meta.graphs : []
      els.graphSelect.innerHTML = ''
      graphs.forEach(graph => {
        if (!graph || typeof graph.slug !== 'string') return
        const option = document.createElement('option')
        option.value = graph.slug
        option.textContent = graph.prompt || graph.slug
        els.graphSelect.appendChild(option)
      })

      const urlSlug = util.params.get('slug')
      const slug = preferredSlug || urlSlug || graphs[0]?.slug
      if (!slug) {
        setStatus('No graphs found')
        return
      }
      els.graphSelect.value = slug
      renderGraph(slug)
    } catch (err) {
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
      setStatus(`Uploaded: ${uploaded.slug}`)
      await loadGraphList(uploaded.slug)
      renderGraph(uploaded.slug, { resetGraphState: true })
    } catch (err) {
      renderError(err)
    } finally {
      setBusy(false)
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
      setStatus(`Loading: ${slug}`)
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
      setStatus(currentGraph?.prompt || slug)
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
    els.uploadButton.disabled = isBusy || !selectedUploadFile()
  }

  function setStatus(text) {
    els.status.textContent = text
  }

  function renderError(err) {
    setStatus(err.message || String(err))
  }
})()
