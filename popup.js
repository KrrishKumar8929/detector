const $ = id => document.getElementById(id);
async function refresh() {
  const state = await chrome.runtime.sendMessage({ type: 'GET_STATE' });
  const selectedChannels = state.selectedChannels || [];
  const hasSelected = selectedChannels.length > 0;
  $('status').textContent = !state.isDiscord ? 'Open Discord Web in the active tab.' : !hasSelected ? 'Add one or more channels to enable detection.' : state.enabled ? `Monitoring ${selectedChannels.length} selected channel${selectedChannels.length === 1 ? '' : 's'}` : 'Monitoring stopped';
  $('status').className = 'status' + (!state.isDiscord ? ' bad' : '');
  $('toggle').textContent = state.enabled ? 'Stop monitoring' : 'Start monitoring';
  $('toggle').disabled = !hasSelected;
  const panelPrefs = await chrome.storage.local.get(['detectorPanelsVisible']);
  $('panelVisibility').textContent = panelPrefs.detectorPanelsVisible === false ? 'Show panels on all tiles' : 'Hide panels on all tiles';
  $('chooseChannel').disabled = !state.currentChannel;
  $('currentChannel').textContent = state.currentChannel ? `Current: ${state.currentChannel.name} (ID ${state.currentChannel.channelId})` : 'Open a specific Discord text channel first (not the server or Friends page).';
  renderChannels(selectedChannels);
  $('autoSwitch').checked = state.autoSwitch !== false;
  const prefs = await chrome.storage.local.get(['bot', 'outputMode', 'languageMode', 'selectedLanguage', 'blacklistPokemonNames']);
  if (document.activeElement !== $('blacklistNames')) $('blacklistNames').value = (prefs.blacklistPokemonNames || []).join('\n');
  $('bot').value = prefs.bot || 'bot1';
  $('outputMode').value = state.outputMode || prefs.outputMode || 'command';
  $('languageMode').value = state.languageMode || prefs.languageMode || 'current';
  $('selectedLanguage').value = state.selectedLanguage || prefs.selectedLanguage || 'english';
  $('selectedLanguage').disabled = $('languageMode').value !== 'selected';
  $('copy').textContent = $('outputMode').value === 'name' ? 'Copy Pokémon name' : 'Copy catch command';
  render(state.latest || {});
}
function renderChannels(channels) {
  const container = $('selectedChannel');
  container.innerHTML = '';
  if (!channels.length) {
    container.textContent = 'No channels selected — detection is paused until you add one.';
    return;
  }
  const list = document.createElement('div');
  list.className = 'channel-list';
  for (const channel of channels) {
    const row = document.createElement('div');
    row.className = 'channel-item';
    const label = document.createElement('span');
    label.textContent = `${channel.name || 'Discord channel'} · ${channel.channelId}`;
    const remove = document.createElement('button');
    remove.textContent = 'Remove';
    remove.title = 'Remove this channel';
    remove.addEventListener('click', async () => {
      await chrome.runtime.sendMessage({ type: 'REMOVE_SELECTED_CHANNEL', guildId: channel.guildId, channelId: channel.channelId });
      await refresh();
    });
    row.append(label, remove);
    list.append(row);
  }
  container.append(list);
}
function render(r) {
  $('pokemon').textContent = r.pokemon ? r.pokemon.replaceAll('_',' ') : r.status === 'error' ? 'Recognition error' : 'No Pokémon detected yet.';
  $('score').textContent = typeof r.score === 'number' ? `Similarity score: ${r.score.toFixed(4)} (not a probability)` : '';
  $('languageNames').textContent = (r.languageNames || []).map(item => `${item.label}: ${item.name}`).join(' · ');
  $('command').value = r.command || '';
  $('copy').disabled = !r.command;
  if (r.blacklisted) $('blacklistStatus').textContent = 'Skip-list match: only a single dot (.) is copied.';
  $('error').textContent = r.error || '';
}
$('chooseChannel').addEventListener('click', async () => {
  $('channelError').textContent = '';
  const state = await chrome.runtime.sendMessage({ type: 'GET_STATE' });
  if (!state.currentChannel) { $('channelError').textContent = 'Could not identify a channel. Open a Discord text channel and try again.'; return; }
  const result = await chrome.runtime.sendMessage({ type: 'ADD_SELECTED_CHANNEL', channel: state.currentChannel });
  if (!result?.ok) $('channelError').textContent = result?.error || 'Could not add this channel.';
  await refresh();
});
$('clearChannel').addEventListener('click', async () => {
  await chrome.runtime.sendMessage({ type: 'CLEAR_SELECTED_CHANNELS' });
  await refresh();
});
$('autoSwitch').addEventListener('change', async () => {
  await chrome.runtime.sendMessage({ type: 'SET_AUTO_SWITCH', enabled: $('autoSwitch').checked });
});
$('toggle').addEventListener('click', async () => {
  const state = await chrome.runtime.sendMessage({ type: 'GET_STATE' });
  if (!(state.selectedChannels || []).length) { $('channelError').textContent = 'Add at least one channel first.'; return; }
  await chrome.runtime.sendMessage({ type: 'TOGGLE_MONITORING', enabled: !state.enabled });
  await refresh();
});
$('bot').addEventListener('change', async () => {
  await chrome.runtime.sendMessage({ type: 'SET_BOT', bot: $('bot').value });
  await refresh();
});
$('languageMode').addEventListener('change', async () => {
  await chrome.runtime.sendMessage({ type: 'SET_LANGUAGE_MODE', mode: $('languageMode').value });
  $('selectedLanguage').disabled = $('languageMode').value !== 'selected';
  await refresh();
});
$('selectedLanguage').addEventListener('change', async () => {
  await chrome.runtime.sendMessage({ type: 'SET_LANGUAGE', language: $('selectedLanguage').value });
  await refresh();
});
$('outputMode').addEventListener('change', async () => {
  await chrome.runtime.sendMessage({ type: 'SET_OUTPUT_MODE', mode: $('outputMode').value });
  $('copy').textContent = $('outputMode').value === 'name' ? 'Copy Pokémon name' : 'Copy catch command';
  await refresh();
});
$('copy').addEventListener('click', async () => {
  const value = $('command').value;
  if (!value) return;
  try { await navigator.clipboard.writeText(value); $('copy').textContent = 'Copied!'; }
  catch (_) { $('command').focus(); $('command').select(); document.execCommand('copy'); $('copy').textContent = 'Copy attempted'; }
  setTimeout(() => $('copy').textContent = $('outputMode').value === 'name' ? 'Copy Pokémon name' : 'Copy catch command', 1300);
});
$('saveBlacklist').addEventListener('click', async () => {
  const names = $('blacklistNames').value.split(/[\n,;]+/).map(name => name.trim()).filter(Boolean);
  const result = await chrome.runtime.sendMessage({ type: 'SAVE_POKEMON_BLACKLIST', names });
  $('blacklistStatus').textContent = result?.ok ? `Saved ${result.blacklistPokemonNames.length} Pokémon name${result.blacklistPokemonNames.length === 1 ? '' : 's'}.` : 'Could not save the skip list.';
  await refresh();
});
chrome.runtime.onMessage.addListener(msg => { if (msg.type === 'RESULT_UPDATED') render(msg.latest); });
refresh();

$('panelVisibility').addEventListener('click', async () => { const state = await chrome.storage.local.get(['detectorPanelsVisible']); await chrome.storage.local.set({ detectorPanelsVisible: state.detectorPanelsVisible === false }); await refresh(); });
