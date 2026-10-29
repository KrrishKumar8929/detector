const $ = id => document.getElementById(id);
async function refresh() {
  const state = await chrome.runtime.sendMessage({ type: 'GET_STATE' });
  const hasSelected = !!state.selectedChannel;
  $('status').textContent = !state.isDiscord ? 'Open Discord Web in the active tab.' : !hasSelected ? 'Choose a channel to enable detection.' : state.enabled ? 'Monitoring selected channel only' : 'Monitoring stopped';
  $('status').className = 'status' + (!state.isDiscord ? ' bad' : '');
  $('toggle').textContent = state.enabled ? 'Stop monitoring' : 'Start monitoring';
  $('toggle').disabled = !hasSelected;
  $('chooseChannel').disabled = !state.currentChannel;
  $('currentChannel').textContent = state.currentChannel ? `Current channel ID: ${state.currentChannel.channelId}` : 'Open a specific Discord text channel first (not the server or Friends page).';
  $('selectedChannel').textContent = state.selectedChannel ? `Selected channel ID: ${state.selectedChannel.channelId}` : 'No channel selected — detection is paused until you choose one.';
  $('autoSwitch').checked = state.autoSwitch !== false;
  const prefs = await chrome.storage.local.get(['bot', 'outputMode']);
  $('bot').value = prefs.bot || 'bot1';
  $('outputMode').value = state.outputMode || prefs.outputMode || 'command';
  $('copy').textContent = $('outputMode').value === 'name' ? 'Copy Pokémon name' : 'Copy catch command';
  render(state.latest || {});
}
function render(r) {
  $('pokemon').textContent = r.pokemon ? r.pokemon.replaceAll('_',' ') : r.status === 'error' ? 'Recognition error' : 'No Pokémon detected yet.';
  $('score').textContent = typeof r.score === 'number' ? `Similarity score: ${r.score.toFixed(4)} (not a probability)` : '';
  $('command').value = r.command || '';
  $('copy').disabled = !r.command;
  $('error').textContent = r.error || '';
}
$('chooseChannel').addEventListener('click', async () => {
  $('channelError').textContent = '';
  const state = await chrome.runtime.sendMessage({ type: 'GET_STATE' });
  if (!state.currentChannel) { $('channelError').textContent = 'Could not identify a channel. Open a Discord text channel and try again.'; return; }
  await chrome.runtime.sendMessage({ type: 'SET_SELECTED_CHANNEL', channel: state.currentChannel });
  await refresh();
});
$('clearChannel').addEventListener('click', async () => {
  await chrome.runtime.sendMessage({ type: 'SET_SELECTED_CHANNEL', channel: null });
  await chrome.runtime.sendMessage({ type: 'TOGGLE_MONITORING', enabled: false });
  await refresh();
});
$('autoSwitch').addEventListener('change', async () => {
  await chrome.runtime.sendMessage({ type: 'SET_AUTO_SWITCH', enabled: $('autoSwitch').checked });
});
$('toggle').addEventListener('click', async () => {
  const state = await chrome.runtime.sendMessage({ type: 'GET_STATE' });
  if (!state.selectedChannel) { $('channelError').textContent = 'Choose a channel first.'; return; }
  await chrome.runtime.sendMessage({ type: 'TOGGLE_MONITORING', enabled: !state.enabled });
  await refresh();
});
$('bot').addEventListener('change', async () => {
  await chrome.runtime.sendMessage({ type: 'SET_BOT', bot: $('bot').value });
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
chrome.runtime.onMessage.addListener(msg => { if (msg.type === 'RESULT_UPDATED') render(msg.latest); });
refresh();
