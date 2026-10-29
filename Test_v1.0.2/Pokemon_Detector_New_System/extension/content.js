// Pokémon Detector v1.0 — resilient Discord spawn observer.
// Watches Discord's message tree continuously; image/recognition work is sent to background.js.
let monitoring = false;
let observer = null;
let currentGeneration = 0;
let scanTimer = 0;
const processed = new Set();
const baselineMessageIds = new Set();
const pendingMessages = new Set();
const imageWaitLogged = new Set();
const MESSAGE_SELECTOR = '[id^="chat-messages-"], [data-list-item-id^="chat-messages"], li[id*="chat-messages"]';
const SPAWN_PATTERNS = [
  /a\s+new\s+wild\s+pok[eé]mon\s+has\s+appeared/i,
  /a\s+wild\s+pok[eé]mon\s+has\s+appeared/i,
  /wild\s+pok[eé]mon\s+has\s+appeared/i,
  /guess\s+the\s+pok[eé]mon\s+and\s+type/i,
  /appeared\s+in\s+the\s+wild/i,
  /what\s+pok[eé]mon\s+is\s+this/i,
  /pok[eé]mon\s+(?:spawned|has\s+spawned)/i
];

function extensionContextAvailable() {
  try { return !!(globalThis.chrome && chrome.runtime && chrome.runtime.id); }
  catch (_) { return false; }
}
function safeSendMessage(message, callback) {
  if (!extensionContextAvailable()) return;
  try {
    const result = chrome.runtime.sendMessage(message, response => {
      try { if (chrome.runtime.lastError) return; } catch (_) { return; }
      if (callback) callback(response);
    });
    if (result && typeof result.catch === 'function') result.catch(() => {});
  } catch (_) { /* A reload can invalidate old page scripts; refresh the tab to restore it. */ }
}
function messageId(node) {
  if (!node || node.nodeType !== Node.ELEMENT_NODE) return '';
  return node.id || node.getAttribute('data-list-item-id') || node.getAttribute('data-message-id') || '';
}
function findMessageNodes(root) {
  const found = [];
  if (!root) return found;
  if (root.nodeType === Node.ELEMENT_NODE && root.matches?.(MESSAGE_SELECTOR)) found.push(root);
  if (typeof root.querySelectorAll === 'function') found.push(...root.querySelectorAll(MESSAGE_SELECTOR));
  return found;
}
function nearestMessage(node) {
  let el = node?.nodeType === Node.ELEMENT_NODE ? node : node?.parentElement;
  if (!el) return null;
  const direct = el.closest?.(MESSAGE_SELECTOR);
  if (direct) return direct;
  // Discord occasionally changes the exact wrapper selector; walk a few ancestors as fallback.
  let cur = el;
  for (let i = 0; cur && i < 7; i++, cur = cur.parentElement) {
    if (cur.matches?.('li[data-list-item-id], li[id], [data-message-id]')) return cur;
  }
  return null;
}
function messageText(message) {
  try { return (message.innerText || message.textContent || '').replace(/\s+/g, ' ').trim(); }
  catch (_) { return ''; }
}
function hasSpawnText(message) {
  let cur = message;
  // The spawn wording may live in a child embed or a sibling wrapper, so check the message and a few parents.
  for (let depth = 0; cur && depth < 4; depth++, cur = cur.parentElement) {
    const text = messageText(cur);
    if (text && SPAWN_PATTERNS.some(pattern => pattern.test(text))) return true;
  }
  return false;
}
function isPokemonImageUrl(value) {
  if (!value) return false;
  try {
    const url = new URL(value, location.href);
    if (!/^https?:$/.test(url.protocol)) return false;
    const host = url.hostname.toLowerCase();
    const path = url.pathname.toLowerCase();
    const discordHost = /^(?:cdn\.discordapp\.com|media\.discordapp\.net|images-ext-\d+\.discordapp\.net)$/.test(host);
    if (discordHost && /\/(?:attachments|ephemeral-attachments|external)\//.test(path)) return true;
    // Pokétwo's own CDN can appear as the original embed URL rather than a Discord attachment.
    if (host === 'cdn.poketwo.net' && /\/images\//.test(path)) return true;
    return false;
  } catch (_) { return false; }
}
function chooseSpawnImage(message) {
  const candidates = [];
  const add = (url, element, alt = '') => {
    if (!isPokemonImageUrl(url)) return;
    const width = Number(element?.naturalWidth || element?.width || 0);
    const height = Number(element?.naturalHeight || element?.height || 0);
    // Ignore tiny icons/emoji where dimensions are known; don't reject unloaded images (0x0) yet.
    if (width > 0 && height > 0 && width < 100 && height < 100) return;
    candidates.push({ url: String(url).replace(/&amp;/g, '&'), element, alt, area: width * height });
  };
  for (const img of message.querySelectorAll('img')) {
    const alt = img.alt || img.getAttribute('aria-label') || '';
    // Check every common source field. currentSrc can be empty while Discord has already set src.
    add(img.currentSrc, img, alt);
    add(img.src, img, alt);
    add(img.getAttribute('src'), img, alt);
    add(img.getAttribute('data-src'), img, alt);
    const srcset = img.getAttribute('srcset');
    if (srcset) for (const item of srcset.split(',')) add(item.trim().split(/\s+/)[0], img, img.alt || '');
  }
  for (const anchor of message.querySelectorAll('a[href]')) {
    const href = anchor.href;
    if (/\.(?:jpe?g|png|webp|gif)(?:$|[?#])/i.test(href) || /cdn\.poketwo\.net\/images\//i.test(href)) add(href, anchor, anchor.innerText || '');
  }
  for (const element of message.querySelectorAll('[style*="background-image"]')) {
    const style = element.style.backgroundImage || getComputedStyle(element).backgroundImage || '';
    const match = style.match(/url\(["']?(.*?)["']?\)/i);
    if (match) add(match[1], element, element.getAttribute('aria-label') || '');
  }
  const unique = [...new Map(candidates.map(item => [item.url, item])).values()];
  if (!unique.length) return null;
  unique.sort((a, b) => b.area - a.area);
  return unique.find(item => /wild|pok[eé]mon|pokemon|spawn/i.test(item.alt)) || unique[0];
}
function inspectMessage(message, generation) {
  if (!monitoring || generation !== currentGeneration || !message?.isConnected) return;
  const id = messageId(message);
  if (!id || baselineMessageIds.has(id) || !hasSpawnText(message)) return;
  const image = chooseSpawnImage(message);
  if (!image) {
    // Helpful diagnostic: confirms text detection even if Discord's image URL is not yet available.
    if (!imageWaitLogged.has(id)) {
      imageWaitLogged.add(id);
      console.debug('[Pokémon Detector] Spawn text matched; waiting for a supported image URL', { messageId: id, text: messageText(message).slice(0, 180), imageCount: message.querySelectorAll('img').length });
    }
    return; // Image may be inserted after the text; the observer will retry.
  }
  const key = `${id}|${image.url}`;
  if (processed.has(key)) return;
  processed.add(key);
  imageWaitLogged.delete(id);
  if (processed.size > 4000) processed.clear();
  console.debug('[Pokémon Detector] Spawn candidate detected', { messageId: id, imageUrl: image.url, at: Date.now() });
  safeSendMessage({ type: 'SPAWN_IMAGE', messageId: id, imageUrl: image.url });
}
function queueMessage(message, generation) {
  if (!message || !monitoring || generation !== currentGeneration) return;
  pendingMessages.add(message);
  if (scanTimer) return;
  scanTimer = setTimeout(() => {
    scanTimer = 0;
    const batch = [...pendingMessages]; pendingMessages.clear();
    for (const item of batch) inspectMessage(item, generation);
  }, 15);
}
function processMutationTarget(target, generation) {
  const message = nearestMessage(target);
  if (message) queueMessage(message, generation);
  if (target?.nodeType === Node.ELEMENT_NODE) {
    for (const childMessage of findMessageNodes(target)) queueMessage(childMessage, generation);
  }
}

// Small floating in-page panel. It stays inside Discord and does not send commands.
let detectorPanel = null;
let panelBody = null;
let panelStatus = null;
let panelPokemon = null;
let panelImage = null;
let panelCommand = null;
let panelCollapsed = false;
let localLatest = null;
let panelVisible = true;
try { localLatest = JSON.parse(sessionStorage.getItem('pokemonDetectorLocalLatest') || 'null'); } catch (_) {}
if (extensionContextAvailable()) {
  try { chrome.storage.local.get(['detectorPanelsVisible']).then(v => { panelVisible = v.detectorPanelsVisible !== false; if (detectorPanel) detectorPanel.style.display = panelVisible ? 'block' : 'none'; if (panelVisible) ensureDetectorPanel(); }).catch(() => { panelVisible = true; ensureDetectorPanel(); }); } catch (_) { ensureDetectorPanel(); }
  try { chrome.storage.onChanged.addListener((changes, area) => { if (area === 'local' && changes.detectorPanelsVisible) { panelVisible = changes.detectorPanelsVisible.newValue !== false; if (detectorPanel) detectorPanel.style.display = panelVisible ? 'block' : 'none'; } }); } catch (_) {}
} else { ensureDetectorPanel(); }
function syncPanelState() {
  safeSendMessage({ type: 'GET_STATE' }, response => {
    if (response && response.activeTabId != null && localLatest) updatePanel(localLatest);
    else if (response && response.latest && response.latest.sourceTabId === response.activeTabId) updatePanel(response.latest);
    if (response && response.enabled && panelStatus && (!response.latest || response.latest.status === 'idle')) {
      panelStatus.textContent = 'Monitoring active — waiting for a new spawn';
    }
  });
}

function ensureDetectorPanel() {
  if (detectorPanel || !document.body) return;
  detectorPanel = document.createElement('section');
  detectorPanel.id = 'pokemon-detector-floating-panel';
  detectorPanel.style.cssText = `position:fixed;right:14px;bottom:14px;width:260px;height:auto;min-width:190px;min-height:100px;max-width:80vw;max-height:85vh;resize:both;z-index:2147483647;background:#171923;color:#f7f7fb;border:1px solid #626a83;border-radius:12px;box-shadow:0 6px 24px #0008;font:13px/1.4 system-ui,sans-serif;overflow:auto;display:${panelVisible ? 'block' : 'none'};`;;
  const header = document.createElement('div');
  header.style.cssText = 'display:flex;align-items:center;justify-content:space-between;padding:9px 10px;background:#292e42;font-weight:700;cursor:grab;touch-action:none;user-select:none;';
  header.append('Pokémon Detector');
  const stopAll = document.createElement('button');
  stopAll.textContent = 'Stop'; stopAll.title = 'Stop detector in all tabs and tiles';
  stopAll.style.cssText = 'border:0;border-radius:5px;background:#b83240;color:white;font-size:12px;font-weight:700;padding:4px 8px;cursor:pointer;margin-left:auto;margin-right:6px;';
  stopAll.addEventListener('click', (event) => { event.stopPropagation(); if (panelStatus) panelStatus.textContent = 'Stopping detector in all tiles…'; safeSendMessage({ type: 'STOP_ALL_DETECTION' }, () => { if (panelStatus) panelStatus.textContent = 'Detector stopped in all tiles'; }); });
  const collapse = document.createElement('button');
  collapse.textContent = '×'; collapse.title = 'Hide panels on all tiles';
  collapse.style.cssText = 'border:0;border-radius:5px;background:#414860;color:white;font-size:18px;width:26px;height:24px;cursor:pointer;';
  header.append(stopAll, collapse);
  panelBody = document.createElement('div'); panelBody.style.cssText = 'padding:10px;';
  panelStatus = document.createElement('div'); panelStatus.textContent = 'Monitoring not started'; panelStatus.style.cssText = 'color:#c5cbe0;margin-bottom:7px;';
  panelImage = document.createElement('img'); panelImage.alt = 'Latest Pokémon spawn'; panelImage.style.cssText = 'display:none;width:100%;max-height:145px;object-fit:contain;background:#0e1018;border-radius:7px;margin-bottom:8px;';
  panelPokemon = document.createElement('div'); panelPokemon.textContent = 'Waiting for a new spawn…'; panelPokemon.style.cssText = 'font-weight:700;font-size:15px;overflow-wrap:anywhere;margin-bottom:8px;';
  panelCommand = document.createElement('div'); panelCommand.textContent = 'Catch command will appear here'; panelCommand.style.cssText = 'display:none;';
  const copy = document.createElement('button'); copy.textContent = 'Copy command';
  copy.style.cssText = 'width:100%;border:0;border-radius:7px;padding:8px;background:#6875e8;color:white;font-weight:700;cursor:pointer;';
  copy.addEventListener('click', () => { const command = panelCommand.dataset.command || ''; if (command) copyCommand(command); });
  panelBody.append(panelStatus, panelImage, panelPokemon, panelCommand, copy);
  detectorPanel.append(header, panelBody); document.body.appendChild(detectorPanel);
  syncPanelState();
  collapse.addEventListener('click', () => { if (extensionContextAvailable()) { try { chrome.storage.local.set({ detectorPanelsVisible: false }).catch(() => {}); } catch (_) {} } });
  // Drag the panel by its header; the minimize button remains clickable.
  header.addEventListener('pointerdown', (event) => {
    if (event.target === collapse || collapse.contains(event.target) || event.target === stopAll || stopAll.contains(event.target) || event.button !== 0) return;
    const rect = detectorPanel.getBoundingClientRect();
    const startX = event.clientX, startY = event.clientY;
    const startLeft = rect.left, startTop = rect.top;
    detectorPanel.style.left = `${startLeft}px`;
    detectorPanel.style.top = `${startTop}px`;
    detectorPanel.style.right = 'auto';
    detectorPanel.style.bottom = 'auto';
    header.style.cursor = 'grabbing';
    try { header.setPointerCapture(event.pointerId); } catch (_) {}
    const move = (e) => {
      const left = Math.max(0, Math.min(window.innerWidth - detectorPanel.offsetWidth, startLeft + e.clientX - startX));
      const top = Math.max(0, Math.min(window.innerHeight - Math.min(detectorPanel.offsetHeight, 80), startTop + e.clientY - startY));
      detectorPanel.style.left = `${left}px`;
      detectorPanel.style.top = `${top}px`;
    };
    const stop = () => {
      header.style.cursor = 'grab';
      header.removeEventListener('pointermove', move);
      header.removeEventListener('pointerup', stop);
      header.removeEventListener('pointercancel', stop);
    };
    header.addEventListener('pointermove', move);
    header.addEventListener('pointerup', stop);
    header.addEventListener('pointercancel', stop);
    event.preventDefault();
  });
}
function copyCommand(command) {
  try {
    const textarea = document.createElement('textarea'); textarea.value = command; textarea.setAttribute('readonly', '');
    textarea.style.cssText = 'position:fixed;left:-9999px;top:0;'; document.body.appendChild(textarea); textarea.select();
    const copied = document.execCommand('copy'); textarea.remove();
    if (copied && panelStatus) panelStatus.textContent = 'Command copied — paste with Ctrl+V';
    else if (panelStatus) panelStatus.textContent = 'Copy blocked; try the extension popup';
  } catch (e) { if (panelStatus) panelStatus.textContent = 'Clipboard copy failed'; }
}
function updatePanel(latest) {
  ensureDetectorPanel(); if (!detectorPanel || !latest) return;
  if (latest.sourceTabId != null && localLatest?.sourceTabId != null && latest.sourceTabId !== localLatest.sourceTabId) return;
  if (latest.sourceTabId != null) { localLatest = latest; try { sessionStorage.setItem('pokemonDetectorLocalLatest', JSON.stringify(latest)); } catch (_) {} }
  if (latest.imageUrl && panelImage && (latest.status === 'processing' || latest.status === 'recognized')) {
    if (panelImage.src !== latest.imageUrl) panelImage.src = latest.imageUrl;
    panelImage.style.display = 'block';
    panelImage.onerror = () => { panelImage.style.display = 'none'; console.warn('[Pokémon Detector] Could not display the spawn image in the panel.'); };
  }
  if (latest.status === 'processing') {
    panelStatus.textContent = 'New spawn detected — recognizing…';
    return; // Keep this tile's last recognized Pokémon visible until the new one is identified.
  } else if (latest.status === 'recognized') {
    panelStatus.textContent = latest.blacklisted ? 'Skip-list match — copied only a dot (.)' : 'Recognition complete';
    panelPokemon.textContent = latest.displayName || latest.pokemon || latest.catchName || 'Pokémon recognized';
    panelCommand.textContent = latest.command || 'No command available'; panelCommand.dataset.command = latest.command || '';
  } else if (latest.status === 'error') {
    panelStatus.textContent = 'Recognition error — keeping previous result';
  } else if (latest.status === 'no-match') {
    panelStatus.textContent = 'No confident match — keeping previous result';
  }
}

function start() {
  stop();
  monitoring = true;
  ensureDetectorPanel();
  if (panelStatus) panelStatus.textContent = 'Monitoring active — waiting for a new spawn';
  syncPanelState();
  currentGeneration++;
  const generation = currentGeneration;
  // Baseline only currently-rendered messages; future text/image mutations are still observed.
  for (const node of findMessageNodes(document)) {
    const id = messageId(node);
    if (id) baselineMessageIds.add(id);
  }
  console.info('[Pokémon Detector] Monitoring started; resilient observer active.');
  observer = new MutationObserver(mutations => {
    if (!monitoring || generation !== currentGeneration) return;
    for (const mutation of mutations) {
      processMutationTarget(mutation.target, generation);
      for (const node of mutation.addedNodes || []) processMutationTarget(node, generation);
    }
  });
  observer.observe(document.body || document.documentElement, {
    childList: true, subtree: true, attributes: true, characterData: true,
    attributeFilter: ['src', 'srcset', 'data-src', 'href', 'style', 'alt', 'aria-label']
  });
}
function stop() {
  monitoring = false;
  currentGeneration++;
  if (scanTimer) clearTimeout(scanTimer);
  scanTimer = 0;
  pendingMessages.clear();
  if (observer) observer.disconnect();
  observer = null;
  if (panelStatus) panelStatus.textContent = 'Monitoring paused';
}

if (extensionContextAvailable()) {
  try {
    chrome.runtime.onMessage.addListener((msg) => {
      if (msg?.type === 'START') start();
      else if (msg?.type === 'STOP') stop();
      else if (msg?.type === 'RESULT_UPDATED' && msg.latest) {
        safeSendMessage({ type: 'GET_STATE' }, state => {
          if (msg.latest.sourceTabId === state?.activeTabId) updatePanel(msg.latest);
        });
      }
    });
  } catch (_) {}
}

// Copy recognized catch commands without sending them to Discord.
if (extensionContextAvailable()) try { chrome.runtime.onMessage.addListener(msg => {
  if (msg.type !== 'COPY_COMMAND' || typeof msg.command !== 'string' || !msg.command) return;
  try {
    const textarea = document.createElement('textarea');
    textarea.value = msg.command;
    textarea.setAttribute('readonly', '');
    textarea.style.position = 'fixed';
    textarea.style.opacity = '0';
    document.body.appendChild(textarea);
    textarea.select();
    const copied = document.execCommand('copy');
    textarea.remove();
    if (copied) console.info('[Pokémon Detector] Catch command copied to clipboard.');
    else console.warn('[Pokémon Detector] Automatic clipboard copy was blocked; use Copy command in the popup.');
  } catch (error) {
    console.warn('[Pokémon Detector] Automatic clipboard copy failed:', error);
  }
}); } catch (_) {}


if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', ensureDetectorPanel, { once: true }); else ensureDetectorPanel();
// Discord can replace its body during startup/navigation; retry panel creation briefly.
let panelRetryCount = 0;
const panelRetryTimer = setInterval(() => {
  panelRetryCount++;
  if (detectorPanel && detectorPanel.isConnected) { clearInterval(panelRetryTimer); return; }
  if (panelVisible && document.body) ensureDetectorPanel();
  if (panelRetryCount >= 20) clearInterval(panelRetryTimer);
}, 500);
