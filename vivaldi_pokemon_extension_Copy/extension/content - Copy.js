let monitoring = false;
let observer = null;
let processed = new Set();
let currentGeneration = 0;

function messageId(node) {
  return node.id || node.getAttribute('data-list-item-id') || node.getAttribute('data-message-id') || '';
}
function findMessageNodes(root) {
  const selectors = '[id^="chat-messages-"][id], [data-list-item-id^="chat-messages"], li[id*="chat-messages"]';
  const found = [];
  if (root instanceof Element && root.matches(selectors)) found.push(root);
  if (root.querySelectorAll) found.push(...root.querySelectorAll(selectors));
  return found;
}
function looksLikeSpawn(message) {
  const text = (message.innerText || message.textContent || '').toLowerCase();
  // Keep the filter conservative: an image alone is not enough to classify a message as a spawn.
  return /wild\s+(?:\w+\s+)?pok[eé]mon|a\s+pok[eé]mon\s+has\s+appeared|pok[eé]mon\s+appeared|guess\s+the\s+pok[eé]mon|appeared\s+in\s+the\s+wild/.test(text);
}
function getImageUrl(img) {
  const candidates = [img.currentSrc, img.src, img.getAttribute('data-src')].filter(Boolean);
  return candidates.find(value => /^https:\/\/(?:cdn\.discordapp\.com|media\.discordapp\.net)\//i.test(value)) || '';
}
function inspectMessage(message, generation) {
  if (!monitoring || generation !== currentGeneration || !document.visibilityState || document.visibilityState !== 'visible') return;
  const id = messageId(message);
  if (!id || processed.has(id) || !looksLikeSpawn(message)) return;
  const imgs = [...message.querySelectorAll('img')];
  const url = imgs.map(getImageUrl).find(Boolean);
  if (!url) return; // A later DOM mutation will retry when the URL appears.
  processed.add(id);
  chrome.runtime.sendMessage({ type: 'SPAWN_IMAGE', messageId: id, imageUrl: url }).catch(() => {});
}
function start() {
  stop();
  monitoring = true;
  currentGeneration++;
  const generation = currentGeneration;
  // Baseline current messages so enabling/switching tabs never processes old spawns.
  for (const node of findMessageNodes(document)) {
    const id = messageId(node);
    if (id) processed.add(id);
  }
  observer = new MutationObserver(mutations => {
    if (!monitoring || generation !== currentGeneration || document.visibilityState !== 'visible') return;
    for (const mutation of mutations) {
      for (const node of mutation.addedNodes) {
        if (!(node instanceof Element)) continue;
        for (const message of findMessageNodes(node)) inspectMessage(message, generation);
        // Discord sometimes fills an existing message's attachment after inserting the row.
        const parentMessage = node.closest?.('[id^="chat-messages-"][id], [data-list-item-id^="chat-messages"]');
        if (parentMessage && !processed.has(messageId(parentMessage))) inspectMessage(parentMessage, generation);
      }
      if (mutation.target instanceof Element) {
        const message = mutation.target.closest('[id^="chat-messages-"][id], [data-list-item-id^="chat-messages"]');
        if (message && !processed.has(messageId(message))) inspectMessage(message, generation);
      }
    }
  });
  observer.observe(document.body, { childList: true, subtree: true, attributes: true, attributeFilter: ['src', 'data-src'] });
}
function stop() {
  monitoring = false;
  currentGeneration++;
  if (observer) observer.disconnect();
  observer = null;
}
chrome.runtime.onMessage.addListener(msg => {
  if (msg.type === 'START') start();
  if (msg.type === 'STOP') stop();
});
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState !== 'visible') stop();
});
