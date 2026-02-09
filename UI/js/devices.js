window.DevicesPage = {
  init: function() {
let editingDeviceId = null;

function showMessage(text, type = 'info') {
  const container = document.getElementById('messageContainer');
  container.innerHTML = `<div class="message ${type}">${text}</div>`;
  setTimeout(() => {
    container.innerHTML = '';
  }, 5000);
}

async function loadDevices() {
  try {
    const response = await fetch('/api/devices');
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    const data = await response.json();
    const devices = Array.isArray(data.devices) ? data.devices : [];
    displayDevices(devices);
  } catch (error) {
    showMessage('Failed to load devices: ' + error.message, 'error');
    const list = document.getElementById('deviceList');
    if (list) {
      list.innerHTML = '<p>Error loading devices. Please refresh the page.</p>';
    }
  }
}

function displayDevices(devices) {
  const list = document.getElementById('deviceList');
  if (!list) {
    console.error('deviceList element not found');
    return;
  }
  
  // Ensure devices is an array
  if (!Array.isArray(devices)) {
    devices = [];
  }
  
  if (devices.length === 0) {
    list.innerHTML = '<p>No devices registered. Add a device using the form above.</p>';
    return;
  }
  
  try {
    list.innerHTML = devices.map(device => {
      // Validate device object has required fields
      if (!device || typeof device.id === 'undefined' || !device.mac_address || !device.name) {
        console.warn('Invalid device object:', device);
        return '';
      }
      const id = parseInt(device.id);
      return `
        <div class="device-item" data-device-id="${id}" data-device-mac="${escapeAttr(device.mac_address)}" data-device-name="${escapeAttr(device.name)}">
          <div class="device-header">
            <div>
              <div class="device-name">${escapeHtml(String(device.name))}</div>
              <div class="device-mac">${escapeHtml(String(device.mac_address))}</div>
            </div>
            <div class="device-actions">
              <button type="button" data-action="edit">Edit</button>
              <button type="button" class="danger" data-action="delete">Delete</button>
            </div>
          </div>
        </div>
      `;
    }).filter(html => html !== '').join('');
  } catch (error) {
    console.error('Error displaying devices:', error);
    list.innerHTML = '<p>Error displaying devices. Please refresh the page.</p>';
  }
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

/** Escape for use inside a single-quoted HTML attribute (e.g. onclick) so quotes in names don't break the attribute. */
function escapeAttr(s) {
  return String(s == null ? '' : s)
    .replace(/&/g, '&amp;').replace(/'/g, '&#39;').replace(/"/g, '&quot;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function editDevice(id, mac, name) {
  editingDeviceId = id;
  document.getElementById('deviceId').value = id;
  document.getElementById('macAddress').value = mac;
  document.getElementById('deviceName').value = name;
  document.getElementById('formTitle').textContent = 'Edit Device';
  document.getElementById('cancelBtn').classList.remove('hidden');
  document.getElementById('macAddress').disabled = true; // Don't allow MAC change
  document.getElementById('saveBtn').textContent = 'Update Device';
  document.getElementById('deviceForm').scrollIntoView({ behavior: 'smooth' });
}

function resetForm() {
  editingDeviceId = null;
  document.getElementById('deviceId').value = '';
  document.getElementById('deviceForm').reset();
  document.getElementById('formTitle').textContent = 'Add Device';
  document.getElementById('cancelBtn').classList.add('hidden');
  document.getElementById('macAddress').disabled = false;
  document.getElementById('saveBtn').textContent = 'Save Device';
}

async function saveDevice() {
  const mac = document.getElementById('macAddress').value.trim().toUpperCase();
  const name = document.getElementById('deviceName').value.trim();
  const id = document.getElementById('deviceId').value;

  if (!mac || !name) {
    showMessage('Please fill in all fields', 'error');
    return;
  }

  // Validate MAC address format
  const macPattern = /^([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}$/;
  if (!macPattern.test(mac)) {
    showMessage('Invalid MAC address format. Use format: XX:XX:XX:XX:XX:XX', 'error');
    return;
  }

  try {
    let response;
    if (id) {
      // Update existing device
      response = await fetch(`/api/devices/${id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mac_address: mac, name: name })
      });
    } else {
      // Create new device
      response = await fetch('/api/devices', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mac_address: mac, name: name })
      });
    }

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to save device');
    }

    showMessage(id ? 'Device updated successfully' : 'Device added successfully', 'success');
    resetForm();
    loadDevices();
  } catch (error) {
    showMessage('Failed to save device: ' + error.message, 'error');
  }
}

async function deleteDevice(id, name) {
  if (!confirm(`Are you sure you want to delete device "${name}"?`)) {
    return;
  }

  try {
    const response = await fetch(`/api/devices/${id}`, {
      method: 'DELETE'
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to delete device');
    }

    showMessage('Device deleted successfully', 'success');
    loadDevices();
  } catch (error) {
    showMessage('Failed to delete device: ' + error.message, 'error');
  }
}

async function scanDevices() {
  const scanBtn = document.getElementById('scanBtn');
  const scanResults = document.getElementById('scanResults');
  const scanList = document.getElementById('scanList');

  scanBtn.disabled = true;
  scanBtn.textContent = 'Scanning...';
  scanResults.classList.remove('hidden');
  scanList.innerHTML = '<p>Scanning for devices...</p>';

  try {
    const response = await fetch('/scan');
    const data = await response.json();
    const devices = data.devices || [];

    if (devices.length === 0) {
      scanList.innerHTML = '<p>No devices found. Make sure your Movesense sensor is powered on and nearby.</p>';
    } else {
      scanList.innerHTML = devices.map(device => `
        <div class="scan-item">
          <div>
            <strong>${escapeHtml(device.name || 'Unknown')}</strong><br/>
            <span style="font-family: monospace; font-size: 12px; color: #666;">${escapeHtml(device.address)}</span>
          </div>
          <button onclick="useScannedDevice('${escapeAttr(device.address)}', '${escapeAttr(device.name || device.address)}')">Use</button>
        </div>
      `).join('');
    }
  } catch (error) {
    scanList.innerHTML = '<p style="color: red;">Scan failed: ' + escapeHtml(error.message) + '</p>';
  } finally {
    scanBtn.disabled = false;
    scanBtn.textContent = 'Scan for Devices';
  }
}

function useScannedDevice(mac, name) {
  resetForm();
  document.getElementById('macAddress').value = mac;
  document.getElementById('deviceName').value = name;
  document.getElementById('scanResults').classList.add('hidden');
  document.getElementById('deviceForm').scrollIntoView({ behavior: 'smooth' });
}

// Event listeners
document.getElementById('deviceForm').addEventListener('submit', (e) => {
  e.preventDefault();
  saveDevice();
});

document.getElementById('cancelBtn').addEventListener('click', () => {
  resetForm();
});

document.getElementById('scanBtn').addEventListener('click', scanDevices);

// Delegated click so Edit/Delete work when view is injected by SPA (no reliance on inline onclick)
const deviceListEl = document.getElementById('deviceList');
if (deviceListEl) {
  deviceListEl.addEventListener('click', function (e) {
    const btn = e.target.closest('button[data-action]');
    if (!btn) return;
    const item = e.target.closest('.device-item');
    if (!item) return;
    const id = parseInt(item.getAttribute('data-device-id'), 10);
    const mac = item.getAttribute('data-device-mac') || '';
    const name = item.getAttribute('data-device-name') || '';
    if (btn.getAttribute('data-action') === 'edit') editDevice(id, mac, name);
    else if (btn.getAttribute('data-action') === 'delete') deleteDevice(id, name);
  });
}

// Load devices on page load
loadDevices();

window.editDevice = editDevice;
window.deleteDevice = deleteDevice;
window.useScannedDevice = useScannedDevice;
  }
};
