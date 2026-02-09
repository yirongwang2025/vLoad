window.SkatersPage = {
  init: function() {
vLoad.renderNav('skaters');
let editingSkaterId = null;

function showMessage(text, type = 'info') {
  const container = document.getElementById('messageContainer');
  container.innerHTML = `<div class="message ${type}">${text}</div>`;
  setTimeout(() => {
    container.innerHTML = '';
  }, 5000);
}

async function loadSkaters() {
  try {
    const response = await fetch('/api/skaters');
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    const data = await response.json();
    const skaters = Array.isArray(data.skaters) ? data.skaters : [];
    displaySkaters(skaters);
  } catch (error) {
    showMessage('Failed to load skaters: ' + error.message, 'error');
    const list = document.getElementById('skaterList');
    if (list) {
      list.innerHTML = '<p>Error loading skaters. Please refresh the page.</p>';
    }
  }
}

function displaySkaters(skaters) {
  const list = document.getElementById('skaterList');
  if (!list) {
    console.error('skaterList element not found');
    return;
  }
  
  if (!Array.isArray(skaters)) {
    skaters = [];
  }
  
  if (skaters.length === 0) {
    list.innerHTML = '<p>No skaters registered. Add a skater using the form above.</p>';
    return;
  }
  
  try {
    list.innerHTML = skaters.map(skater => {
      if (!skater || typeof skater.id === 'undefined' || !skater.name) {
        console.warn('Invalid skater object:', skater);
        return '';
      }
      const details = [];
      if (skater.level) details.push(`Level: ${escapeHtml(skater.level)}`);
      if (skater.club) details.push(`Club: ${escapeHtml(skater.club)}`);
      if (skater.date_of_birth) details.push(`DOB: ${escapeHtml(skater.date_of_birth.split('T')[0])}`);
      const sid = parseInt(skater.id);
      return `
        <div class="skater-item" data-skater-id="${sid}" data-skater-name="${escapeAttr(skater.name)}" data-skater-dob="${escapeAttr(skater.date_of_birth || '')}" data-skater-gender="${escapeAttr(skater.gender || '')}" data-skater-level="${escapeAttr(skater.level || '')}" data-skater-club="${escapeAttr(skater.club || '')}" data-skater-email="${escapeAttr(skater.email || '')}" data-skater-phone="${escapeAttr(skater.phone || '')}" data-skater-notes="${escapeAttr((skater.notes || '').replace(/'/g, "\\'"))}">
          <div class="skater-header">
            <div>
              <div class="skater-name">${escapeHtml(skater.name)}</div>
              <div class="skater-details">${details.join(' • ')}</div>
            </div>
            <div class="skater-actions">
              <button type="button" data-action="edit">Edit</button>
              <button type="button" data-action="relationships">Manage Relationships</button>
              <button type="button" class="danger" data-action="delete">Delete</button>
            </div>
          </div>
        </div>
      `;
    }).filter(html => html !== '').join('');
  } catch (error) {
    console.error('Error displaying skaters:', error);
    list.innerHTML = '<p>Error displaying skaters. Please refresh the page.</p>';
  }
}

function escapeHtml(text) {
  if (!text) return '';
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

/** Escape for use inside a single-quoted HTML attribute (data-*) so quotes don't break. */
function escapeAttr(s) {
  return String(s == null ? '' : s)
    .replace(/&/g, '&amp;').replace(/'/g, '&#39;').replace(/"/g, '&quot;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function editSkater(id, name, dob, gender, level, club, email, phone, notes) {
  editingSkaterId = id;
  document.getElementById('skaterId').value = id;
  document.getElementById('skaterName').value = name;
  document.getElementById('dateOfBirth').value = dob ? dob.split('T')[0] : '';
  document.getElementById('gender').value = gender || '';
  document.getElementById('level').value = level || '';
  document.getElementById('club').value = club || '';
  document.getElementById('email').value = email || '';
  document.getElementById('phone').value = phone || '';
  document.getElementById('notes').value = notes || '';
  document.getElementById('formTitle').textContent = 'Edit Skater';
  document.getElementById('cancelBtn').classList.remove('hidden');
  document.getElementById('saveBtn').textContent = 'Update Skater';
  document.getElementById('skaterForm').scrollIntoView({ behavior: 'smooth' });
}

function resetForm() {
  editingSkaterId = null;
  document.getElementById('skaterId').value = '';
  document.getElementById('skaterForm').reset();
  document.getElementById('formTitle').textContent = 'Add Skater';
  document.getElementById('cancelBtn').classList.add('hidden');
  document.getElementById('saveBtn').textContent = 'Save Skater';
}

async function saveSkater() {
  const name = document.getElementById('skaterName').value.trim();
  const dob = document.getElementById('dateOfBirth').value;
  const gender = document.getElementById('gender').value;
  const level = document.getElementById('level').value.trim();
  const club = document.getElementById('club').value.trim();
  const email = document.getElementById('email').value.trim();
  const phone = document.getElementById('phone').value.trim();
  const notes = document.getElementById('notes').value.trim();
  const id = document.getElementById('skaterId').value;

  if (!name) {
    showMessage('Please fill in the name field', 'error');
    return;
  }

  try {
    let response;
    const payload = {
      name: name,
      date_of_birth: dob || null,
      gender: gender || null,
      level: level || null,
      club: club || null,
      email: email || null,
      phone: phone || null,
      notes: notes || null
    };
    
    if (id) {
      response = await fetch(`/api/skaters/${id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
    } else {
      response = await fetch('/api/skaters', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
    }

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to save skater');
    }

    showMessage(id ? 'Skater updated successfully' : 'Skater added successfully', 'success');
    resetForm();
    loadSkaters();
  } catch (error) {
    showMessage('Failed to save skater: ' + error.message, 'error');
  }
}

async function deleteSkater(id, name) {
  if (!confirm(`Are you sure you want to delete skater "${name}"?`)) {
    return;
  }

  try {
    const response = await fetch(`/api/skaters/${id}`, {
      method: 'DELETE'
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to delete skater');
    }

    showMessage('Skater deleted successfully', 'success');
    loadSkaters();
  } catch (error) {
    showMessage('Failed to delete skater: ' + error.message, 'error');
  }
}

// Event listeners
const skaterFormEl = document.getElementById('skaterForm');
if (skaterFormEl) skaterFormEl.addEventListener('submit', (e) => { e.preventDefault(); saveSkater(); });

const cancelBtnEl = document.getElementById('cancelBtn');
if (cancelBtnEl) cancelBtnEl.addEventListener('click', () => { resetForm(); });

// Delegated click for relationship modal (Remove buttons in coachesList/devicesList)
const relationshipModalEl = document.getElementById('relationshipModal');
if (relationshipModalEl) {
  relationshipModalEl.addEventListener('click', function (e) {
    const btn = e.target.closest('button[data-action]');
    if (!btn) return;
    const action = btn.getAttribute('data-action');
    if (action === 'remove-coach') {
      const coachId = parseInt(btn.getAttribute('data-coach-id'), 10);
      if (coachId) removeCoachFromSkater(coachId);
    } else if (action === 'remove-device') {
      const deviceId = parseInt(btn.getAttribute('data-device-id'), 10);
      if (deviceId) removeDeviceFromSkater(deviceId);
    }
  });
}

// Delegated click for list buttons (same pattern as Devices - works with SPA-injected content)
const skaterListEl = document.getElementById('skaterList');
if (skaterListEl) {
  skaterListEl.addEventListener('click', function (e) {
    const btn = e.target.closest('button[data-action]');
    if (!btn) return;
    const item = e.target.closest('.skater-item');
    if (!item) return;
    const id = parseInt(item.getAttribute('data-skater-id'), 10);
    const name = item.getAttribute('data-skater-name') || '';
    const dob = item.getAttribute('data-skater-dob') || '';
    const gender = item.getAttribute('data-skater-gender') || '';
    const level = item.getAttribute('data-skater-level') || '';
    const club = item.getAttribute('data-skater-club') || '';
    const email = item.getAttribute('data-skater-email') || '';
    const phone = item.getAttribute('data-skater-phone') || '';
    const notes = item.getAttribute('data-skater-notes') || '';
    const action = btn.getAttribute('data-action');
    if (action === 'edit') editSkater(id, name, dob, gender, level, club, email, phone, notes);
    else if (action === 'relationships') openRelationshipModal(id, name);
    else if (action === 'delete') deleteSkater(id, name);
  });
}

// Relationship management
let currentSkaterId = null;
let allCoaches = [];
let allDevices = [];

async function loadCoaches() {
  try {
    const response = await fetch('/api/coaches');
    const data = await response.json();
    allCoaches = Array.isArray(data.coaches) ? data.coaches : [];
  } catch (error) {
    console.error('Failed to load coaches:', error);
  }
}

async function loadDevices() {
  try {
    const response = await fetch('/api/devices');
    const data = await response.json();
    allDevices = Array.isArray(data.devices) ? data.devices : [];
  } catch (error) {
    console.error('Failed to load devices:', error);
  }
}

async function openRelationshipModal(skaterId, skaterName) {
  currentSkaterId = skaterId;
  document.getElementById('modalTitle').textContent = `Manage Relationships: ${escapeHtml(skaterName)}`;
  const modal = document.getElementById('relationshipModal');
  modal.classList.remove('hidden');
  modal.style.display = 'flex';
  
  // Populate dropdowns
  const coachSelect = document.getElementById('coachSelect');
  coachSelect.innerHTML = '<option value="">-- Select Coach --</option>';
  allCoaches.forEach(coach => {
    const option = document.createElement('option');
    option.value = coach.id;
    option.textContent = coach.name;
    coachSelect.appendChild(option);
  });

  const deviceSelect = document.getElementById('deviceSelect');
  deviceSelect.innerHTML = '<option value="">-- Select Device --</option>';
  allDevices.forEach(device => {
    const option = document.createElement('option');
    option.value = device.id;
    option.textContent = `${device.name} (${device.mac_address})`;
    deviceSelect.appendChild(option);
  });

  await loadSkaterRelationships(skaterId);
}

function closeRelationshipModal() {
  const modal = document.getElementById('relationshipModal');
  modal.classList.add('hidden');
  modal.style.display = 'none';
  currentSkaterId = null;
}

async function loadSkaterRelationships(skaterId) {
  try {
    const response = await fetch(`/api/skaters/${skaterId}`);
    if (!response.ok) throw new Error('Failed to load skater');
    const skater = await response.json();
    
    // Display coaches
    const coachesList = document.getElementById('coachesList');
    if (skater.coaches && skater.coaches.length > 0) {
      coachesList.innerHTML = skater.coaches.map(c => `
        <div style="padding: 8px; background: #f5f5f5; margin-bottom: 4px; border-radius: 4px; display: flex; justify-content: space-between; align-items: center;">
          <span>${escapeHtml(c.coach_name)} ${c.is_head_coach ? '<strong>(Head Coach)</strong>' : ''}</span>
          <button type="button" data-action="remove-coach" data-coach-id="${c.coach_id}" style="padding: 4px 8px; font-size: 12px;">Remove</button>
        </div>
      `).join('');
    } else {
      coachesList.innerHTML = '<p style="color: #666; font-size: 13px;">No coaches assigned</p>';
    }

    // Display devices
    const devicesList = document.getElementById('devicesList');
    if (skater.devices && skater.devices.length > 0) {
      devicesList.innerHTML = skater.devices.map(d => `
        <div style="padding: 8px; background: #f5f5f5; margin-bottom: 4px; border-radius: 4px; display: flex; justify-content: space-between; align-items: center;">
          <span>${escapeHtml(d.device_name)} (${escapeHtml(d.mac_address)}) - <strong>${escapeHtml(d.placement)}</strong></span>
          <button type="button" data-action="remove-device" data-device-id="${d.device_id}" style="padding: 4px 8px; font-size: 12px;">Remove</button>
        </div>
      `).join('');
    } else {
      devicesList.innerHTML = '<p style="color: #666; font-size: 13px;">No devices assigned</p>';
    }
  } catch (error) {
    showMessage('Failed to load relationships: ' + error.message, 'error');
  }
}

async function addCoachToSkater() {
  const coachId = parseInt(document.getElementById('coachSelect').value);
  const isHeadCoach = document.getElementById('headCoachCheck').checked;
  if (!coachId || !currentSkaterId) return;

  try {
    const response = await fetch(`/api/skaters/${currentSkaterId}/coaches`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ coach_id: coachId, is_head_coach: isHeadCoach })
    });
    if (!response.ok) throw new Error('Failed to add coach');
    document.getElementById('coachSelect').value = '';
    document.getElementById('headCoachCheck').checked = false;
    await loadSkaterRelationships(currentSkaterId);
    showMessage('Coach added successfully', 'success');
  } catch (error) {
    showMessage('Failed to add coach: ' + error.message, 'error');
  }
}

async function removeCoachFromSkater(coachId) {
  if (!confirm('Remove this coach?')) return;
  try {
    const response = await fetch(`/api/skaters/${currentSkaterId}/coaches/${coachId}`, {
      method: 'DELETE'
    });
    if (!response.ok) throw new Error('Failed to remove coach');
    await loadSkaterRelationships(currentSkaterId);
    showMessage('Coach removed successfully', 'success');
  } catch (error) {
    showMessage('Failed to remove coach: ' + error.message, 'error');
  }
}

async function addDeviceToSkater() {
  const deviceId = parseInt(document.getElementById('deviceSelect').value);
  const placement = document.getElementById('placementSelect').value;
  if (!deviceId || !currentSkaterId) return;

  try {
    const response = await fetch(`/api/skaters/${currentSkaterId}/devices`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ device_id: deviceId, placement: placement })
    });
    if (!response.ok) throw new Error('Failed to add device');
    document.getElementById('deviceSelect').value = '';
    document.getElementById('placementSelect').value = 'waist';
    await loadSkaterRelationships(currentSkaterId);
    showMessage('Device added successfully', 'success');
  } catch (error) {
    showMessage('Failed to add device: ' + error.message, 'error');
  }
}

async function removeDeviceFromSkater(deviceId) {
  if (!confirm('Remove this device?')) return;
  try {
    const response = await fetch(`/api/skaters/${currentSkaterId}/devices/${deviceId}`, {
      method: 'DELETE'
    });
    if (!response.ok) throw new Error('Failed to remove device');
    await loadSkaterRelationships(currentSkaterId);
    showMessage('Device removed successfully', 'success');
  } catch (error) {
    showMessage('Failed to remove device: ' + error.message, 'error');
  }
}

// Load skaters, coaches, and devices on page load
loadSkaters();
loadCoaches();
loadDevices();

// Expose modal handlers for inline onclick (Add, Close buttons in static modal HTML)
window.closeRelationshipModal = closeRelationshipModal;
window.addCoachToSkater = addCoachToSkater;
window.addDeviceToSkater = addDeviceToSkater;
  }
};
