// Discord DOM observer. Processes only new spawn messages while this tab is active.
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

function getDiscordImageUrl(img) {
  const candidates = [img.currentSrc, img.src, img.getAttribute('data-src')].filter(Boolean);
  return candidates.find(value => /^https:\/\/(?:cdn\.discordapp\.com|media\.discordapp\.net)\/attachments\//i.test(value)) || '';
}

function chooseSpawnImage(message) {
  const imgs = [...message.querySelectorAll('img')];
  // Prefer actual attachment images. Discord avatars/emojis use /avatars/ and /emojis/.
  const attachmentImages = imgs
    .map(img => ({ img, url: getDiscordImageUrl(img) }))
    .filter(item => item.url);
  if (!attachmentImages.length) return null;

  // The Pokétwo spawn image normally has alt text containing the spawn announcement.
  const altMatch = attachmentImages.find(({ img }) => /wild|pok[eé]mon|pokemon/i.test(img.alt || ''));
  if (altMatch) return altMatch;

  // Otherwise use a named image attachment, never a Discord avatar or emoji.
  const namedAttachment = attachmentImages.find(({ url }) => /\/pokemon\.(?:jpe?g|png|webp)(?:[?#]|$)/i.test(url));
  return namedAttachment || attachmentImages[0];
}

function inspectMessage(message, generation) {
  if (!monitoring || generation !== currentGeneration || document.visibilityState !== 'visible') return;
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

function start() {
  stop();
  monitoring = true;
  currentGeneration++;
  const generation = currentGeneration;

  // Baseline current messages so starting/switching tabs does not process old spawns.
  for (const node of findMessageNodes(document)) {
    const id = messageId(node);
    if (id) processed.add(id);
  }
  console.debug('[Pokémon Detector] Monitoring started on active tab; baseline messages:', processed.size);

  observer = new MutationObserver(mutations => {
    if (!monitoring || generation !== currentGeneration || document.visibilityState !== 'visible') return;
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

