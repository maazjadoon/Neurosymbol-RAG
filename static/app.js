/* app.js — Neurosymbol RAG frontend
 * ─────────────────────────────────
 * Pure vanilla JS, no framework required.
 * All API calls go to the FastAPI backend at the same origin.
 */

'use strict';

/* ── Config ─────────────────────────────────────────────────────────────── */
const API_BASE = '';          // same-origin; empty string = relative URLs
const MAX_EXCERPT_CHARS = 280;

/* ── State ──────────────────────────────────────────────────────────────── */
let lastQuery = '';
let selectedFile = null;
let verifiedValue = false;

// Neurosymbolic State
let nsEnabled = false;
let activeWorkflow = '';
let sessionId = '';

let WORKFLOWS = {};

async function fetchWorkflows() {
  try {
    const resp = await fetch(`${API_BASE}/workflows`);
    if (resp.ok) {
      WORKFLOWS = await resp.json();
      populateWorkflowSelect();
    }
  } catch (err) {
    console.error("Failed to fetch workflows from backend:", err);
  }
}

function populateWorkflowSelect() {
  const select = $('ns-workflow-select');
  if (!select) return;
  
  // Keep the first "None" option
  select.innerHTML = '<option value="">None</option>';
  
  for (const wfName of Object.keys(WORKFLOWS)) {
    const opt = document.createElement('option');
    opt.value = wfName;
    const displayName = wfName.replace(/_/g, ' ');
    opt.textContent = displayName;
    select.appendChild(opt);
  }
}

function parseMarkdown(text) {
  if (!text) return '';
  // Simple markdown parser converting headers, bold, inline code, and lists to HTML safely.
  // We escape HTML first to prevent XSS.
  let escaped = escapeHtml(text);
  
  // Replace headings: ### text -> <h3>text</h3>
  escaped = escaped.replace(/###\s+(.*?)(<br\s*\/?>|$)/g, '<h3>$1</h3>$2');
  
  // Replace bold: **text** -> <strong>text</strong>
  escaped = escaped.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  
  // Replace lists: 1. text -> list item
  escaped = escaped.replace(/(\d+)\.\s+(.*?)(<br\s*\/?>|$)/g, '<div class="md-list-item"><strong>$1.</strong> $2</div>$3');
  
  // Replace inline code: `text` -> <code>text</code>
  escaped = escaped.replace(/`(.*?)`/g, '<code>$1</code>');
  
  // Replace paragraph break
  escaped = escaped.replace(/\n\n/g, '<br/><br/>');
  escaped = escaped.replace(/\n/g, '<br/>');
  
  return escaped;
}

function toggleNeurosymbolic(enabled) {
  nsEnabled = enabled;
  const wrapper = $('ns-workflow-wrapper');
  if (enabled) {
    if (wrapper) {
      wrapper.style.opacity = '1';
      wrapper.style.pointerEvents = 'auto';
    }
    if (!sessionId) {
      sessionId = 'sess_' + Math.random().toString(36).substring(2, 11);
    }
    show('btn-clear-session');
  } else {
    if (wrapper) {
      wrapper.style.opacity = '0.5';
      wrapper.style.pointerEvents = 'none';
    }
    hide('btn-clear-session');
  }
}

function handleWorkflowChange(val) {
  activeWorkflow = val;
}

async function clearActiveSession() {
  if (!sessionId) return;
  try {
    const submitBtn = $('btn-clear-session');
    if (submitBtn) submitBtn.disabled = true;
    
    const resp = await fetch(`${API_BASE}/clear-session?session_id=${encodeURIComponent(sessionId)}`, {
      method: 'POST'
    });
    if (resp.ok) {
      console.log(`Session ${sessionId} cleared`);
    }
  } catch (err) {
    console.error('Failed to clear session on backend', err);
  } finally {
    const submitBtn = $('btn-clear-session');
    if (submitBtn) submitBtn.disabled = false;
  }
  
  // Generate a new session ID
  sessionId = 'sess_' + Math.random().toString(36).substring(2, 11);
  
  // Clear UI displays
  const badgesContainer = $('symptom-badges');
  if (badgesContainer) {
    badgesContainer.innerHTML = '<span class="no-features-text">No concepts detected yet.</span>';
  }
  const pathsContainer = $('path-tracer');
  if (pathsContainer) {
    pathsContainer.innerHTML = '<span class="no-paths-text">No path traversed yet.</span>';
  }
  $('complexity-display').textContent = '0.000';
  $('modulation-display').textContent = '0.000';
  
  const insightCard = $('llm-insight-card');
  if (insightCard) {
    insightCard.hidden = true;
    insightCard.style.display = 'none';
    $('llm-insight-body').innerHTML = '';
  }
  
  // Re-run search if there was a previous search
  if (lastQuery) {
    handleSearch(null);
  }
}

/* ── Utility ─────────────────────────────────────────────────────────────── */
function $(id) { return document.getElementById(id); }

function show(id) {
  const el = $(id);
  if (el) {
    el.hidden = false;
    el.style.display = '';
  }
}

function hide(id) {
  const el = $(id);
  if (el) {
    el.hidden = true;
    el.style.display = 'none';
  }
}

function fmt(n) {
  return typeof n === 'number' ? n.toFixed(3) : '—';
}

function pct(n) {
  return Math.min(100, Math.max(0, (n || 0) * 100)).toFixed(1);
}

function excerpt(text, maxLen = MAX_EXCERPT_CHARS) {
  if (!text) return '';
  const trimmed = text.trim().replace(/\s+/g, ' ');
  return trimmed.length > maxLen
    ? trimmed.slice(0, maxLen).replace(/\s+\S*$/, '') + '…'
    : trimmed;
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1048576).toFixed(1)} MB`;
}

function domainLabel(domain) {
  const map = { tech: 'Technology', legal: 'Legal', health: 'Health', business: 'Business' };
  return map[domain] || domain;
}

function domainIcon(domain) {
  const icons = {
    tech:     `<svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>`,
    legal:    `<svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/></svg>`,
    health:   `<svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>`,
    business: `<svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>`,
  };
  return icons[domain] || '';
}

/* ── Tab switching ───────────────────────────────────────────────────────── */
function switchTab(tab) {
  const tabs    = ['search', 'ingest'];
  const navBtns = { search: $('nav-search'), ingest: $('nav-ingest') };
  const panels  = { search: $('panel-search'), ingest: $('panel-ingest') };

  tabs.forEach(t => {
    const isActive = t === tab;
    navBtns[t].classList.toggle('active', isActive);
    navBtns[t].setAttribute('aria-selected', isActive ? 'true' : 'false');
    panels[t].hidden = !isActive;
  });

  // Focus the first interactive element in the activated panel
  const panel = panels[tab];
  const firstFocusable = panel.querySelector('input, button, select, [tabindex="0"]');
  if (firstFocusable) requestAnimationFrame(() => firstFocusable.focus({ preventScroll: true }));
}

/* ── API health check ────────────────────────────────────────────────────── */
async function checkApiStatus() {
  const badge = $('api-status');
  const label = badge.querySelector('.status-label');

  try {
    const resp = await fetch(`${API_BASE}/search?q=test`, { method: 'GET', signal: AbortSignal.timeout(4000) });
    if (resp.ok || resp.status === 422) {
      badge.className = 'status-badge online';
      label.textContent = 'API online';
    } else {
      throw new Error('non-ok');
    }
  } catch {
    badge.className = 'status-badge offline';
    label.textContent = 'API offline';
  }
}

/* ── Filter chip rendering ───────────────────────────────────────────────── */
function inferFilters(q) {
  const lower = q.toLowerCase();
  const chips = [];

  const domains = {
    tech:     ['machine learning', 'deep learning', 'cloud', 'edge ai', 'ai', 'cybersecurity'],
    legal:    ['law', 'legal', 'act', 'regulation', 'compliance'],
    business: ['business', 'market', 'startup', 'funding', 'marketing'],
    health:   ['health', 'medical', 'disease', 'nutrition', 'exercise'],
  };

  for (const [domain, keywords] of Object.entries(domains)) {
    if (keywords.some(kw => lower.includes(kw))) {
      chips.push({ type: 'domain', label: domainLabel(domain) });
      break;
    }
  }

  if (lower.includes('verified'))      chips.push({ type: 'verified', label: '✓ Verified only' });
  if (lower.includes('last 6 months')) chips.push({ type: 'date',     label: 'Last 6 months' });

  return chips;
}

function renderFilterChips(q) {
  const container = $('filter-chips');
  const chips = inferFilters(q);
  container.innerHTML = '';

  chips.forEach(chip => {
    const el = document.createElement('span');
    el.className = `filter-chip ${chip.type}`;
    el.textContent = chip.label;
    container.appendChild(el);
  });
}

/* ── Search flow ─────────────────────────────────────────────────────────── */
function setQuery(text) {
  $('search-input').value = text;
  renderFilterChips(text);
  $('search-form').requestSubmit();
}

function setSearchState(state) {
  const states = ['state-idle', 'state-loading', 'state-error', 'state-empty', 'results-list'];
  states.forEach(s => {
    const el = $(s);
    if (el) {
      el.hidden = true;
      el.style.display = 'none';
    }
  });
  const activeEl = $(state);
  if (activeEl) {
    activeEl.hidden = false;
    activeEl.style.display = '';
  }
}

async function handleSearch(event) {
  if (event) event.preventDefault();

  const q = $('search-input').value.trim();
  if (!q) {
    $('search-input').focus();
    return;
  }

  lastQuery = q;
  renderFilterChips(q);
  setSearchState('state-loading');

  const submitBtn = $('search-form').querySelector('.search-button');
  submitBtn.disabled = true;

  try {
    let url = `${API_BASE}/search?q=${encodeURIComponent(q)}`;
    if (nsEnabled && sessionId) {
      url += `&session_id=${encodeURIComponent(sessionId)}`;
    }
    if (nsEnabled && activeWorkflow) {
      url += `&workflow=${encodeURIComponent(activeWorkflow)}`;
    }
    
    const resp = await fetch(url);

    if (!resp.ok) {
      throw new Error(`Server returned ${resp.status}`);
    }

    const data = await resp.json();
    renderResults(data, q);
  } catch (err) {
    $('error-message').textContent = err.message.includes('fetch')
      ? 'Could not connect to the API. Make sure the FastAPI server is running on this origin.'
      : `Search failed: ${err.message}`;
    setSearchState('state-error');
  } finally {
    submitBtn.disabled = false;
  }
}

function retryLastSearch() {
  if (lastQuery) {
    $('search-input').value = lastQuery;
    handleSearch(null);
  }
}

function renderResults(data, query) {
  // Handle both response shapes:
  //   List (success with results)        → [{doc, bm25_score, vector_score, final_score, why}, ...]
  //   Object with message (no results)   → {results: [], message: "..."}
  const isList    = Array.isArray(data);
  const results   = isList ? data : (data.results || []);
  const nsData    = isList ? null : data.neurosymbolic;
  const llmAnswer = isList ? null : data.llm_answer;

  const dashboard = $('neurosymbolic-dashboard');
  const layout = $('search-results-layout');
  const insightCard = $('llm-insight-card');

  if (llmAnswer) {
    if (insightCard) {
      insightCard.hidden = false;
      insightCard.style.display = '';
      $('llm-insight-body').innerHTML = parseMarkdown(llmAnswer);
    }
  } else {
    if (insightCard) {
      insightCard.hidden = true;
      insightCard.style.display = 'none';
      $('llm-insight-body').innerHTML = '';
    }
  }

  if (nsData) {
    // 1. Update session details
    $('session-id-display').textContent = nsData.session_id || '—';
    $('session-id-display').title = nsData.session_id || '';
    $('complexity-display').textContent = fmt(nsData.complexity);
    $('modulation-display').textContent = fmt(nsData.modulation_strength);

    // 2. Render accumulated features (symptom badges)
    const symptomBadgesContainer = $('symptom-badges');
    symptomBadgesContainer.innerHTML = '';
    const features = nsData.accumulated_features || [];
    if (features.length === 0) {
      symptomBadgesContainer.innerHTML = '<span class="no-features-text">No concepts detected yet.</span>';
    } else {
      features.forEach(f => {
        const span = document.createElement('span');
        span.className = 'symptom-badge';
        span.textContent = f;
        symptomBadgesContainer.appendChild(span);
      });
    }

    // 3. Render traversed nodes with PageRank
    const pathTracerContainer = $('path-tracer');
    pathTracerContainer.innerHTML = '';
    const traversed = nsData.traversed_nodes || [];
    const prScores = nsData.pagerank_scores || {};
    
    // Sort traversed nodes by PR score descending
    const sortedNodes = [...traversed].sort((a, b) => (prScores[b] || 0) - (prScores[a] || 0));

    if (sortedNodes.length === 0) {
      pathTracerContainer.innerHTML = '<span class="no-paths-text">No path traversed yet.</span>';
    } else {
      sortedNodes.forEach(node => {
        const score = prScores[node] || 0;
        const percentage = pct(score);
        
        const nodeRow = document.createElement('div');
        nodeRow.className = 'path-node-row';
        nodeRow.innerHTML = `
          <div class="path-node-info">
            <span class="path-node-name">${escapeHtml(node)}</span>
            <span class="path-node-score">${fmt(score)}</span>
          </div>
          <div class="path-node-bar-track">
            <div class="path-node-bar-fill" style="width: ${percentage}%"></div>
          </div>
        `;
        pathTracerContainer.appendChild(nodeRow);
      });
    }

    // 4. Render clinical workflow checklist if active
    const workflowSection = $('dashboard-workflow-section');
    const stepsList = $('workflow-steps-list');
    
    if (activeWorkflow && WORKFLOWS[activeWorkflow]) {
      show('dashboard-workflow-section');
      stepsList.innerHTML = '';
      
      const steps = WORKFLOWS[activeWorkflow];
      
      // Find the earliest step matched by the documents in results
      let minDocStep = 999;
      results.forEach(item => {
        if (item.workflow_step !== undefined && item.workflow_step < minDocStep) {
          minDocStep = item.workflow_step;
        }
      });
      
      steps.forEach(step => {
        const isCompleted = features.includes(step.concept);
        
        let isActive = false;
        if (minDocStep !== 999) {
          isActive = step.seq === minDocStep;
        } else {
          // Find first incomplete step
          const firstIncomplete = steps.find(s => !features.includes(s.concept));
          isActive = firstIncomplete && step.seq === firstIncomplete.seq;
        }
        
        const li = document.createElement('li');
        
        let statusClass = 'pending';
        let statusIcon = `<span class="step-number">${step.seq}</span>`;
        
        if (isCompleted) {
          statusClass = 'completed';
          statusIcon = `<svg class="step-icon" width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3">
            <polyline points="20 6 9 17 4 12"/>
          </svg>`;
        } else if (isActive) {
          statusClass = 'active';
          statusIcon = `<div class="step-pulse"></div>`;
        }
        
        li.className = `workflow-step-item ${statusClass}`;
        li.innerHTML = `
          <div class="step-status-indicator">${statusIcon}</div>
          <div class="step-content">
            <span class="step-title">${escapeHtml(step.title)}</span>
            <span class="step-concept-badge">${escapeHtml(step.concept)}</span>
          </div>
        `;
        stepsList.appendChild(li);
      });
    } else {
      hide('dashboard-workflow-section');
    }

    // Show dashboard
    if (dashboard) {
      dashboard.hidden = false;
      dashboard.style.display = '';
    }
    if (layout) {
      layout.classList.add('ns-active');
    }
  } else {
    // Hide dashboard
    if (dashboard) {
      dashboard.hidden = true;
      dashboard.style.display = 'none';
    }
    if (layout) {
      layout.classList.remove('ns-active');
    }
  }

  if (results.length === 0) {
    if (nsData) {
      $('results-meta').innerHTML = `<span>0 results for "<strong>${escapeHtml(query)}</strong>"</span>`;
      $('result-cards').innerHTML = `<li class="state-empty" style="padding: var(--space-8) 0; grid-column: span 2; text-align: center; list-style: none;">
        <div class="empty-icon" aria-hidden="true" style="margin-bottom: var(--space-2); color: var(--color-text-muted);">
          <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
            <polyline points="14 2 14 8 20 8"/>
          </svg>
        </div>
        <p class="state-body" style="margin-bottom: 0;">No documents matched your filters. Try broadening your query.</p>
      </li>`;
      setSearchState('results-list');
      show('results-list');
    } else {
      setSearchState('state-empty');
    }
    return;
  }

  // Meta line
  $('results-meta').innerHTML =
    `<span>${results.length} result${results.length !== 1 ? 's' : ''} for "<strong>${escapeHtml(query)}</strong>"</span>`;

  // Build cards
  const list = $('result-cards');
  list.innerHTML = '';

  const template = document.getElementById('result-card-template');

  results.forEach((item, idx) => {
    const card = template.content.cloneNode(true).querySelector('.result-card');
    card.style.animationDelay = `${idx * 40}ms`;

    // Rank badge
    card.querySelector('.result-rank').textContent = `#${idx + 1}`;

    // Title
    card.querySelector('.result-title').textContent = item.doc?.title || 'Untitled';

    // Domain + verified + year badges
    const badgesEl = card.querySelector('.result-badges');
    if (item.doc?.domain) {
      const b = document.createElement('span');
      b.className = `badge badge-${item.doc.domain}`;
      b.innerHTML = `${domainIcon(item.doc.domain)} ${domainLabel(item.doc.domain)}`;
      badgesEl.appendChild(b);
    }
    if (item.doc?.verified) {
      const b = document.createElement('span');
      b.className = 'badge badge-verified';
      b.innerHTML = `<svg width="9" height="9" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="20 6 9 17 4 12"/></svg> Verified`;
      badgesEl.appendChild(b);
    }
    if (item.doc?.year) {
      const b = document.createElement('span');
      b.className = 'badge badge-year';
      b.textContent = item.doc.year;
      badgesEl.appendChild(b);
    }

    // Excerpt
    card.querySelector('.result-excerpt').textContent = excerpt(item.doc?.content);

    // Score bars — animate to final width after paint
    const finalPct  = pct(item.final_score);
    const bm25Pct   = pct(item.bm25_score);
    const vectorPct = pct(item.vector_score);

    const finalFill  = card.querySelector('.score-bar-fill.final');
    const bm25Fill   = card.querySelector('.score-bar-fill.bm25');
    const vectorFill = card.querySelector('.score-bar-fill.vector');

    const tracks = card.querySelectorAll('.score-bar-track');
    if (tracks[0]) {
      tracks[0].setAttribute('aria-valuenow', finalPct);
      tracks[0].setAttribute('aria-label', `Final score ${finalPct}%`);
    }
    if (tracks[1]) {
      tracks[1].setAttribute('aria-valuenow', bm25Pct);
      tracks[1].setAttribute('aria-label', `BM25 score ${bm25Pct}%`);
    }
    if (tracks[2]) {
      tracks[2].setAttribute('aria-valuenow', vectorPct);
      tracks[2].setAttribute('aria-label', `Vector score ${vectorPct}%`);
    }

    card.querySelector('.final-val').textContent  = fmt(item.final_score);
    card.querySelector('.bm25-val').textContent   = fmt(item.bm25_score);
    card.querySelector('.vector-val').textContent = fmt(item.vector_score);

    // Defer width setting so CSS transition fires
    requestAnimationFrame(() => {
      finalFill.style.width  = `${finalPct}%`;
      bm25Fill.style.width   = `${bm25Pct}%`;
      vectorFill.style.width = `${vectorPct}%`;
    });

    // Why reasons
    const whyEl = card.querySelector('.result-why');
    if (item.why && item.why.length > 0) {
      item.why.forEach(reason => {
        const tag = document.createElement('span');
        tag.className = 'why-tag';
        tag.innerHTML = `<svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="20 6 9 17 4 12"/></svg> ${escapeHtml(reason)}`;
        whyEl.appendChild(tag);
      });
    }

    list.appendChild(card);
  });

  setSearchState('results-list');
  show('results-list');
}

/* ── Ingest flow ─────────────────────────────────────────────────────────── */
function handleFileSelect(event) {
  const file = event.target.files?.[0];
  if (file) applyFile(file);
}

function handleDragOver(event) {
  event.preventDefault();
  $('dropzone').classList.add('drag-over');
}

function handleDragLeave(event) {
  $('dropzone').classList.remove('drag-over');
}

function handleDrop(event) {
  event.preventDefault();
  $('dropzone').classList.remove('drag-over');
  const file = event.dataTransfer?.files?.[0];
  if (file && file.type === 'application/pdf') {
    applyFile(file);
  } else if (file) {
    showIngestError('Only PDF files are supported.');
  }
}

function applyFile(file) {
  selectedFile = file;
  $('dropzone-content').hidden = true;

  const preview = $('dropzone-file-preview');
  preview.hidden = false;
  $('file-name-display').textContent = file.name;
  $('file-size-display').textContent = formatBytes(file.size);

  // Reset feedback
  hide('ingest-success');
  hide('ingest-error');
}

function clearFile(event) {
  event.stopPropagation();
  event.preventDefault();
  selectedFile = null;
  $('file-input').value = '';
  $('dropzone-file-preview').hidden = true;
  $('dropzone-content').hidden = false;
  $('dropzone').classList.remove('drag-over');
}

function toggleVerified(btn) {
  verifiedValue = !verifiedValue;
  btn.setAttribute('aria-checked', verifiedValue ? 'true' : 'false');
}

function showIngestError(msg) {
  $('ingest-error-message').textContent = msg;
  show('ingest-error');
  hide('ingest-success');
  hide('ingest-progress');
}

async function handleIngest(event) {
  event.preventDefault();

  if (!selectedFile) {
    showIngestError('Please select a PDF file first.');
    return;
  }

  const domain = $('domain-select').value;
  if (!domain) {
    $('domain-select').focus();
    showIngestError('Please select a domain.');
    return;
  }

  // File size guard (20 MB)
  if (selectedFile.size > 20 * 1024 * 1024) {
    showIngestError('File exceeds the 20 MB limit.');
    return;
  }

  const year = parseInt($('year-input').value, 10);

  // Show progress
  hide('ingest-success');
  hide('ingest-error');
  show('ingest-progress');
  $('progress-label').textContent = 'Uploading and processing…';
  $('ingest-submit').disabled = true;

  const formData = new FormData();
  formData.append('file',     selectedFile);
  formData.append('domain',   domain);
  formData.append('verified', verifiedValue.toString());
  formData.append('year',     year.toString());

  try {
    const resp = await fetch(`${API_BASE}/ingest`, {
      method: 'POST',
      body:   formData,
    });

    if (!resp.ok) {
      const body = await resp.json().catch(() => ({}));
      throw new Error(body.detail || `Server error ${resp.status}`);
    }

    const data = await resp.json();
    hide('ingest-progress');

    $('success-message').textContent = data.message || 'Document ingested successfully.';
    show('ingest-success');

    // Reset form state after success
    clearFile({ stopPropagation: () => {}, preventDefault: () => {} });
    $('domain-select').value = '';
    if (verifiedValue) toggleVerified($('verified-toggle'));

  } catch (err) {
    hide('ingest-progress');
    showIngestError(err.message || 'Ingest failed. Check that the server is running.');
  } finally {
    $('ingest-submit').disabled = false;
  }
}

/* ── XSS guard ───────────────────────────────────────────────────────────── */
function escapeHtml(str) {
  if (typeof str !== 'string') return '';
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

/* ── Live filter chip update on input ───────────────────────────────────── */
$('search-input').addEventListener('input', function () {
  renderFilterChips(this.value);
});

/* ── Keyboard shortcut: / to focus search ────────────────────────────────── */
document.addEventListener('keydown', function (e) {
  const tag = document.activeElement?.tagName;
  if (e.key === '/' && tag !== 'INPUT' && tag !== 'TEXTAREA' && tag !== 'SELECT') {
    e.preventDefault();
    switchTab('search');
    $('search-input').focus();
  }
  if (e.key === 'Escape' && document.activeElement === $('search-input')) {
    $('search-input').blur();
  }
});

/* ── Boot ────────────────────────────────────────────────────────────────── */
checkApiStatus();
toggleNeurosymbolic(false);
fetchWorkflows();
