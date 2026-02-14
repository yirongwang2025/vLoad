window.CoachesPage = {
  init: function() {
vLoad.renderNav('coaches');
let editingCoachId = null;

function showMessage(text, type = 'info') {
  const container = document.getElementById('messageContainer');
  container.innerHTML = `<div class="message ${type}">${text}</div>`;
  setTimeout(() => {
    container.innerHTML = '';
  }, 5000);
}

async function loadCoaches() {
  try {
    const response = await fetch('/api/coaches');
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    const data = await response.json();
    const coaches = Array.isArray(data.coaches) ? data.coaches : [];
    displayCoaches(coaches);
  } catch (error) {
    showMessage('Failed to load coaches: ' + error.message, 'error');
    const list = document.getElementById('coachList');
    if (list) {
      list.innerHTML = '<p>Error loading coaches. Please refresh the page.</p>';
    }
  }
}

function displayCoaches(coaches) {
  const list = document.getElementById('coachList');
  if (!list) {
    console.error('coachList element not found');
    return;
  }
  
  if (!Array.isArray(coaches)) {
    coaches = [];
  }
  
  if (coaches.length === 0) {
    list.innerHTML = '<p>No coaches registered. Add a coach using the form above.</p>';
    return;
  }
  
  try {
    list.innerHTML = coaches.map(coach => {
      if (!coach || typeof coach.id === 'undefined' || !coach.name) {
        console.warn('Invalid coach object:', coach);
        return '';
      }
      const details = [];
      if (coach.certification) details.push(`Cert: ${escapeHtml(coach.certification)}`);
      if (coach.level) details.push(`Level: ${escapeHtml(coach.level)}`);
      if (coach.club) details.push(`Club: ${escapeHtml(coach.club)}`);
      const cid = parseInt(coach.id);
      return `
        <div class="coach-item" data-coach-id="${cid}" data-coach-name="${escapeAttr(coach.name)}" data-coach-email="${escapeAttr(coach.email || '')}" data-coach-phone="${escapeAttr(coach.phone || '')}" data-coach-certification="${escapeAttr(coach.certification || '')}" data-coach-level="${escapeAttr(coach.level || '')}" data-coach-club="${escapeAttr(coach.club || '')}" data-coach-notes="${escapeAttr((coach.notes || '').replace(/'/g, "\\'"))}">
          <div class="coach-header">
            <div>
              <div class="coach-name">${escapeHtml(coach.name)}</div>
              <div class="coach-details">${details.join(' • ')}</div>
            </div>
            <div class="coach-actions">
              <button type="button" data-action="edit">Edit</button>
              <button type="button" data-action="relationships">Manage Skaters</button>
              <button type="button" class="danger" data-action="delete">Delete</button>
            </div>
          </div>
        </div>
      `;
    }).filter(html => html !== '').join('');
  } catch (error) {
    console.error('Error displaying coaches:', error);
    list.innerHTML = '<p>Error displaying coaches. Please refresh the page.</p>';
  }
}

function escapeHtml(text) {
  if (!text) return '';
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

function escapeAttr(s) {
  return String(s == null ? '' : s)
    .replace(/&/g, '&amp;').replace(/'/g, '&#39;').replace(/"/g, '&quot;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function editCoach(id, name, email, phone, certification, level, club, notes) {
  editingCoachId = id;
  document.getElementById('coachId').value = id;
  document.getElementById('coachName').value = name;
  document.getElementById('email').value = email || '';
  document.getElementById('phone').value = phone || '';
  document.getElementById('certification').value = certification || '';
  document.getElementById('level').value = level || '';
  document.getElementById('club').value = club || '';
  document.getElementById('notes').value = notes || '';
  document.getElementById('formTitle').textContent = 'Edit Coach';
  document.getElementById('cancelBtn').classList.remove('hidden');
  document.getElementById('saveBtn').textContent = 'Update Coach';
  document.getElementById('coachForm').scrollIntoView({ behavior: 'smooth' });
}

function resetForm() {
  editingCoachId = null;
  document.getElementById('coachId').value = '';
  document.getElementById('coachForm').reset();
  document.getElementById('formTitle').textContent = 'Add Coach';
  document.getElementById('cancelBtn').classList.add('hidden');
  document.getElementById('saveBtn').textContent = 'Save Coach';
}

async function saveCoach() {
  const name = document.getElementById('coachName').value.trim();
  const email = document.getElementById('email').value.trim();
  const phone = document.getElementById('phone').value.trim();
  const certification = document.getElementById('certification').value.trim();
  const level = document.getElementById('level').value.trim();
  const club = document.getElementById('club').value.trim();
  const notes = document.getElementById('notes').value.trim();
  const id = document.getElementById('coachId').value;

  if (!name) {
    showMessage('Please fill in the name field', 'error');
    return;
  }

  try {
    let response;
    const payload = {
      name: name,
      email: email || null,
      phone: phone || null,
      certification: certification || null,
      level: level || null,
      club: club || null,
      notes: notes || null
    };
    
    if (id) {
      response = await fetch(`/api/coaches/${id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
    } else {
      response = await fetch('/api/coaches', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
    }

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to save coach');
    }

    showMessage(id ? 'Coach updated successfully' : 'Coach added successfully', 'success');
    resetForm();
    loadCoaches();
  } catch (error) {
    showMessage('Failed to save coach: ' + error.message, 'error');
  }
}

async function deleteCoach(id, name) {
  if (!confirm(`Are you sure you want to delete coach "${name}"?`)) {
    return;
  }

  try {
    const response = await fetch(`/api/coaches/${id}`, {
      method: 'DELETE'
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to delete coach');
    }

    showMessage('Coach deleted successfully', 'success');
    loadCoaches();
  } catch (error) {
    showMessage('Failed to delete coach: ' + error.message, 'error');
  }
}

// Event listeners
const coachFormEl = document.getElementById('coachForm');
if (coachFormEl) coachFormEl.addEventListener('submit', (e) => { e.preventDefault(); saveCoach(); });

const cancelBtnEl = document.getElementById('cancelBtn');
if (cancelBtnEl) cancelBtnEl.addEventListener('click', () => { resetForm(); });

// Delegated click for list buttons (same pattern as Devices)
const coachListEl = document.getElementById('coachList');
if (coachListEl) {
  coachListEl.addEventListener('click', function (e) {
    const btn = e.target.closest('button[data-action]');
    if (!btn) return;
    const item = e.target.closest('.coach-item');
    if (!item) return;
    const id = parseInt(item.getAttribute('data-coach-id'), 10);
    const name = item.getAttribute('data-coach-name') || '';
    const email = item.getAttribute('data-coach-email') || '';
    const phone = item.getAttribute('data-coach-phone') || '';
    const certification = item.getAttribute('data-coach-certification') || '';
    const level = item.getAttribute('data-coach-level') || '';
    const club = item.getAttribute('data-coach-club') || '';
    const notes = item.getAttribute('data-coach-notes') || '';
    const action = btn.getAttribute('data-action');
    if (action === 'edit') editCoach(id, name, email, phone, certification, level, club, notes);
    else if (action === 'relationships') openRelationshipModal(id, name);
    else if (action === 'delete') deleteCoach(id, name);
  });
}

// Delegated click for relationship modal (Remove buttons in skatersList)
const relationshipModalEl = document.getElementById('relationshipModal');
if (relationshipModalEl) {
  relationshipModalEl.addEventListener('click', function (e) {
    const btn = e.target.closest('button[data-action]');
    if (!btn) return;
    if (btn.getAttribute('data-action') === 'remove-skater') {
      const skaterId = parseInt(btn.getAttribute('data-skater-id'), 10);
      if (skaterId) removeSkaterFromCoach(skaterId);
    }
  });
}

// Relationship management
let currentCoachId = null;
let allSkaters = [];

async function loadSkaters() {
  try {
    const response = await fetch('/api/skaters');
    const data = await response.json();
    allSkaters = Array.isArray(data.skaters) ? data.skaters : [];
  } catch (error) {
    console.error('Failed to load skaters:', error);
  }
}

async function openRelationshipModal(coachId, coachName) {
  currentCoachId = coachId;
  document.getElementById('modalTitle').textContent = `Manage Skaters: ${escapeHtml(coachName)}`;
  const modal = document.getElementById('relationshipModal');
  modal.classList.remove('hidden');
  modal.style.display = 'flex';
  
  // Populate skater dropdown
  const skaterSelect = document.getElementById('skaterSelect');
  skaterSelect.innerHTML = '<option value="">-- Select Skater --</option>';
  allSkaters.forEach(skater => {
    const option = document.createElement('option');
    option.value = skater.id;
    option.textContent = skater.name;
    skaterSelect.appendChild(option);
  });

  await loadCoachRelationships(coachId);
}

function closeRelationshipModal() {
  const modal = document.getElementById('relationshipModal');
  modal.classList.add('hidden');
  modal.style.display = 'none';
  currentCoachId = null;
}

async function loadCoachRelationships(coachId) {
  try {
    const response = await fetch(`/api/coaches/${coachId}`);
    if (!response.ok) throw new Error('Failed to load coach');
    const coach = await response.json();
    
    // Display skaters
    const skatersList = document.getElementById('skatersList');
    if (coach.skaters && coach.skaters.length > 0) {
      skatersList.innerHTML = coach.skaters.map(s => `
        <div style="padding: 8px; background: #f5f5f5; margin-bottom: 4px; border-radius: 4px; display: flex; justify-content: space-between; align-items: center;">
          <span>${escapeHtml(s.skater_name)} ${s.is_head_coach ? '<strong>(Head Coach)</strong>' : ''}</span>
          <button type="button" data-action="remove-skater" data-skater-id="${s.skater_id}" style="padding: 4px 8px; font-size: 12px;">Remove</button>
        </div>
      `).join('');
    } else {
      skatersList.innerHTML = '<p style="color: #666; font-size: 13px;">No skaters assigned</p>';
    }
  } catch (error) {
    showMessage('Failed to load relationships: ' + error.message, 'error');
  }
}

async function addSkaterToCoach() {
  const skaterId = parseInt(document.getElementById('skaterSelect').value);
  const isHeadCoach = document.getElementById('headCoachCheck').checked;
  if (!skaterId || !currentCoachId) return;

  try {
    const response = await fetch(`/api/coaches/${currentCoachId}/skaters`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ skater_id: skaterId, is_head_coach: isHeadCoach })
    });
    if (!response.ok) throw new Error('Failed to add skater');
    document.getElementById('skaterSelect').value = '';
    document.getElementById('headCoachCheck').checked = false;
    await loadCoachRelationships(currentCoachId);
    showMessage('Skater added successfully', 'success');
  } catch (error) {
    showMessage('Failed to add skater: ' + error.message, 'error');
  }
}

async function removeSkaterFromCoach(skaterId) {
  if (!confirm('Remove this skater?')) return;
  try {
    const response = await fetch(`/api/coaches/${currentCoachId}/skaters/${skaterId}`, {
      method: 'DELETE'
    });
    if (!response.ok) throw new Error('Failed to remove skater');
    await loadCoachRelationships(currentCoachId);
    showMessage('Skater removed successfully', 'success');
  } catch (error) {
    showMessage('Failed to remove skater: ' + error.message, 'error');
  }
}

// Load coaches and skaters on page load
loadCoaches();
loadSkaters();

// Expose modal handlers for inline onclick (Add, Close buttons in static modal HTML)
window.closeRelationshipModal = closeRelationshipModal;
window.addSkaterToCoach = addSkaterToCoach;
  }
};
