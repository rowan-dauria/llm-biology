;(function () {
  const els = {
    graphSelect: document.querySelector('#graph-select'),
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

  disableEditOnlyCircuitTracerControls()

  const graphView = createGraphView(d3.select('.nav'), d3.select('#graph'))

  els.graphSelect.addEventListener('change', () => {
    const slug = els.graphSelect.value
    if (slug) renderGraph(slug, { resetGraphState: true })
  })

  loadGraphList()

  async function loadGraphList(preferredSlug) {
    try {
      const meta = await getJson('./data/graph-metadata.json')
      const graphs = Array.isArray(meta.graphs) ? meta.graphs : []
      els.graphSelect.innerHTML = ''
      graphs.forEach(graph => {
        if (!graph || typeof graph.slug !== 'string') return
        const option = document.createElement('option')
        option.value = graph.slug
        option.textContent = graph.slug
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

  function renderGraph(slug, { resetGraphState = false } = {}) {
    if (resetGraphState) clearGraphStateParams()
    util.params.set('slug', slug)
    graphView.show(slug)
  }

  function clearGraphStateParams() {
    graphStateParamKeys.forEach(key => util.params.set(key, null))
  }

  function disableEditOnlyCircuitTracerControls() {
    const initButtonContainer = window.initCgButtonContainer
    if (typeof initButtonContainer !== 'function') return

    window.initCgButtonContainer = function (args) {
      initButtonContainer(args)
      if (args?.visState?.isEditMode) return

      args.cgSel
        .selectAll('.button-container .toggle-buttons')
        .filter(function () {
          return d3.select(this).text().trim() === 'Ungroup selected'
        })
        .remove()
    }
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
        isEditMode: false,
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

  async function getJson(url) {
    const res = await fetch(url)
    const data = await res.json().catch(() => ({}))
    if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`)
    return data
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

  function setStatus(text) {
    els.status.textContent = text
  }

  function renderError(err) {
    setStatus(err.message || String(err))
  }
})()
