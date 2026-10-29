// Discord DOM observer. Monitors this Discord tab even when it is in the background.
let monitoring = false;
let observer = null;
const processed = new Set();
let currentGeneration = 0;

const MESSAGE_SELECTOR = '[id^="chat-messages-"][id], [data-list-item-id^="chat-messages"], li[id*="chat-messages"]';

function messageId(node) {
  return node.id || node.getAttribute('data-list-item-id') || node.getAttribute('data-message-id') || '';
}

function findMessageNodes(root) {
  const found = [];
  if (root instanceof Element && root.matches(MESSAGE_SELECTOR)) found.push(root);
  if (root && typeof root.querySelectorAll === 'function') found.push(...root.querySelectorAll(MESSAGE_SELECTOR));
  return found;
}

function messageText(message) {
  return (message.innerText || message.textContent || '').replace(/\s+/g, ' ').trim();
}

function looksLikeSpawn(message) {
  const text = messageText(message).toLowerCase();
  // Pokétwo's actual embed can say: "Wild Wiglett fled. A new wild pokémon has appeared!"
  return /a\s+new\s+wild\s+pok[eé]mon\s+has\s+appeared/.test(text)
    || /a\s+wild\s+pok[eé]mon\s+has\s+appeared/.test(text)
    || /wild\s+pok[eé]mon\s+has\s+appeared/.test(text)
    || /guess\s+the\s+pok[eé]mon\s+and\s+type/.test(text)
    || /appeared\s+in\s+the\s+wild/.test(text);
}

function isDiscordAttachmentUrl(value) {
  if (!value) return false;
  try {
    const url = new URL(value, location.href);
    return /^https?:$/.test(url.protocol)
      && /^(?:cdn\.discordapp\.com|media\.discordapp\.net)$/i.test(url.hostname)
      && /\/(?:attachments|ephemeral-attachments)\//i.test(url.pathname);
  } catch (_) { return false; }
}

function chooseSpawnImage(message) {
  // Discord can render an attachment as an <img>, a linked image, or a CSS background.
  const candidates = [];
  for (const img of message.querySelectorAll('img')) {
    for (const value of [img.currentSrc, img.src, img.getAttribute('data-src'), img.getAttribute('srcset')?.split(',')[0]?.trim().split(/\s+/)[0]]) {
      if (isDiscordAttachmentUrl(value)) candidates.push({ url: value, element: img, alt: img.alt || '' });
    }
  }
  for (const anchor of message.querySelectorAll('a[href]')) {
    const value = anchor.href;
    if (isDiscordAttachmentUrl(value) && /\.(?:jpe?g|png|webp|gif)(?:$|[?#])/i.test(value)) {
      candidates.push({ url: value, element: anchor, alt: anchor.innerText || '' });
    }
  }
  for (const element of message.querySelectorAll('[style*="background-image"]')) {
    const style = element.style.backgroundImage || getComputedStyle(element).backgroundImage || '';
    const match = style.match(/url\(["']?(.*?)["']?\)/i);
    if (match && isDiscordAttachmentUrl(match[1])) candidates.push({ url: match[1], element, alt: element.getAttribute('aria-label') || '' });
  }
  const unique = [...new Map(candidates.map(item => [item.url, item])).values()];
  if (!unique.length) return null;
  // Prefer image elements whose accessible text identifies them as the Pokémon spawn.
  const altMatch = unique.find(item => /wild|pok[eé]mon|pokemon|spawn/i.test(item.alt));
  if (altMatch) return altMatch;
  const namedAttachment = unique.find(item => /\/(?:pokemon|spawn)\.(?:jpe?g|png|webp)(?:[?#]|$)/i.test(item.url));
  return namedAttachment || unique[0];
}

function inspectMessage(message, generation) {
  if (!monitoring || generation !== currentGeneration) return;
  const id = messageId(message);
  if (!id || processed.has(id) || !looksLikeSpawn(message)) return;

  const image = chooseSpawnImage(message);
  if (!image) return; // Mutations to the image src will trigger another inspection.

  // Mark only once an attachment URL exists, so a later DOM update can retry if needed.
  processed.add(id);
  console.debug('[Pokémon Detector] New spawn image found', { messageId: id, imageUrl: image.url });
  chrome.runtime.sendMessage({ type: 'SPAWN_IMAGE', messageId: id, imageUrl: image.url })
    .catch(error => console.warn('[Pokémon Detector] Could not contact extension worker:', error));
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
chrome.storage.local.get(['detectorPanelsVisible']).then(v => { panelVisible = v.detectorPanelsVisible !== false; if (detectorPanel) detectorPanel.style.display = panelVisible ? 'block' : 'none'; });
chrome.storage.onChanged.addListener((changes, area) => { if (area === 'local' && changes.detectorPanelsVisible) { panelVisible = changes.detectorPanelsVisible.newValue !== false; if (detectorPanel) detectorPanel.style.display = panelVisible ? 'block' : 'none'; } });
function syncPanelState() {
  try {
    chrome.runtime.sendMessage({ type: 'GET_STATE' }, (response) => {
      if (chrome.runtime.lastError) {
        console.debug('[Pokémon Detector] Panel state sync unavailable:', chrome.runtime.lastError.message);
        return;
      }
      if (response && response.activeTabId != null && localLatest) updatePanel(localLatest);
      else if (response && response.latest && response.latest.sourceTabId === response.activeTabId) updatePanel(response.latest);
      if (response && response.enabled && panelStatus && (!response.latest || response.latest.status === 'idle')) {
        panelStatus.textContent = 'Monitoring active — waiting for a new spawn';
      }
    });
  } catch (error) {
    console.debug('[Pokémon Detector] Panel state sync failed:', error);
  }
}

function ensureDetectorPanel() {
  if (detectorPanel || !document.body) return;
  detectorPanel = document.createElement('section');
  detectorPanel.id = 'pokemon-detector-floating-panel';
  detectorPanel.style.cssText = `position:fixed;right:14px;bottom:14px;width:260px;height:auto;min-width:190px;min-height:100px;max-width:80vw;max-height:85vh;resize:both;z-index:2147483647;background:#171923;color:#f7f7fb;border:1px solid #626a83;border-radius:12px;box-shadow:0 6px 24px #0008;font:13px/1.4 system-ui,sans-serif;overflow:auto;display:${panelVisible ? 'block' : 'none'};`;;
  const header = document.createElement('div');
  header.style.cssText = 'display:flex;align-items:center;justify-content:space-between;padding:9px 10px;background:#292e42;font-weight:700;cursor:grab;touch-action:none;user-select:none;';
  header.append('Pokémon Detector');
  const collapse = document.createElement('button');
  collapse.textContent = '×'; collapse.title = 'Hide panels on all tiles';
  collapse.style.cssText = 'border:0;border-radius:5px;background:#414860;color:white;font-size:18px;width:26px;height:24px;cursor:pointer;';
  header.appendChild(collapse);
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
  collapse.addEventListener('click', async () => { await chrome.storage.local.set({ detectorPanelsVisible: false }); });
  // Drag the panel by its header; the minimize button remains clickable.
  header.addEventListener('pointerdown', (event) => {
    if (event.target === collapse || collapse.contains(event.target) || event.button !== 0) return;
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

  // Baseline current messages so starting/switching tabs does not process old spawns.
  for (const node of findMessageNodes(document)) {
    const id = messageId(node);
    if (id) processed.add(id);
  }
  console.debug('[Pokémon Detector] Monitoring started on this Discord tab (background tabs supported); baseline messages:', processed.size);

  observer = new MutationObserver(mutations => {
    if (!monitoring || generation !== currentGeneration) return;
    for (const mutation of mutations) {
      for (const node of mutation.addedNodes) {
        if (!(node instanceof Element)) continue;
        for (const message of findMessageNodes(node)) inspectMessage(message, generation);
        const parentMessage = node.closest?.(MESSAGE_SELECTOR);
        if (parentMessage && !processed.has(messageId(parentMessage))) inspectMessage(parentMessage, generation);
      }
      if (mutation.target instanceof Element) {
        const message = mutation.target.closest(MESSAGE_SELECTOR);
        if (message && !processed.has(messageId(message))) inspectMessage(message, generation);
      }
    }
  });
  observer.observe(document.body, {
    childList: true, subtree: true, attributes: true,
    attributeFilter: ['src', 'srcset', 'data-src', 'alt']
  });
}

function stop() {
  monitoring = false;
  if (panelStatus) panelStatus.textContent = 'Monitoring paused';
  currentGeneration++;
  if (observer) observer.disconnect();
  observer = null;
}

chrome.runtime.onMessage.addListener(msg => {
  if (msg.type === 'START') start();
  if (msg.type === 'STOP') stop();
});

chrome.runtime.onMessage.addListener(msg => {
  if (msg.type === 'RESULT_UPDATED' && msg.latest) {
    chrome.runtime.sendMessage({ type: 'GET_STATE' }, state => {
      if (chrome.runtime.lastError) return;
      if (msg.latest.sourceTabId === state?.activeTabId) updatePanel(msg.latest);
    });
  }
});

// Copy recognized catch commands without sending them to Discord.
chrome.runtime.onMessage.addListener(msg => {
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
});


if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', ensureDetectorPanel, { once: true }); else ensureDetectorPanel();


// Desktop-panel click support: focus the Discord message composer only; never sends.
chrome.runtime.onMessage.addListener((message) => {
  if (message?.type !== "DESKTOP_PANEL_FOCUS_DISCORD") return;
  const selectors = [
    '[role="textbox"][data-slate-editor="true"]',
    '[role="textbox"][contenteditable="true"]',
    'div[contenteditable="true"][role="textbox"]'
  ];
  for (const selector of selectors) {
    const element = document.querySelector(selector);
    if (element) { element.focus(); break; }
  }
});

