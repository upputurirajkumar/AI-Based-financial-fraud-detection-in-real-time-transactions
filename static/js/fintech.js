/**
 * ENTERPRISE FINTECH API CLIENT & INTERACTION ENGINE
 * Handles REST API calls, toast notifications, interactive modals,
 * inline alert triaging, sample transaction loaders, and chart integrations.
 */

function getCookie(name) {
  let cookieValue = null;
  if (document.cookie && document.cookie !== '') {
    const cookies = document.cookie.split(';');
    for (let i = 0; i < cookies.length; i++) {
      const cookie = cookies[i].trim();
      if (cookie.substring(0, name.length + 1) === (name + '=')) {
        cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
        break;
      }
    }
  }
  return cookieValue;
}

const CSRF_TOKEN = getCookie('csrftoken');

// Centralized REST API Client
const api = {
  async request(endpoint, options = {}) {
    const headers = {
      'Content-Type': 'application/json',
      'X-CSRFToken': CSRF_TOKEN,
      ...(options.headers || {})
    };

    try {
      const res = await fetch(endpoint, { credentials: 'same-origin', ...options, headers });
      const data = await res.json();
      if (!res.ok || (data.success === false)) {
        throw new Error(data.error?.message || `API Error: ${res.statusText}`);
      }
      return data;
    } catch (err) {
      console.error(`API Call failed [${endpoint}]:`, err);
      throw err;
    }
  },

  // Analytics
  getOverview() { return this.request('/api/analytics/overview/'); },
  getRiskDistribution() { return this.request('/api/analytics/risk-distribution/'); },
  getTimeSeries(days = 7, interval = 'daily') { return this.request(`/api/analytics/time-series/?days=${days}&interval=${interval}`); },
  getModels() { return this.request('/api/analytics/models/'); },

  // Transactions
  getTransactions(params = {}) {
    const query = new URLSearchParams(params).toString();
    return this.request(`/api/transactions/?${query}`);
  },
  getTransaction(id) { return this.request(`/api/transactions/${id}/`); },
  ingestTransaction(data) {
    return this.request('/api/transactions/ingest/', {
      method: 'POST',
      body: JSON.stringify(data)
    });
  },

  // Alerts
  getAlerts(params = {}) {
    const query = new URLSearchParams(params).toString();
    return this.request(`/api/alerts/?${query}`);
  },
  transitionAlert(alertId, status, notes = '') {
    return this.request(`/api/alerts/${alertId}/transition/`, {
      method: 'PATCH',
      body: JSON.stringify({ status, notes })
    });
  },

  // Cases
  getCases(params = {}) {
    const query = new URLSearchParams(params).toString();
    return this.request(`/api/cases/?${query}`);
  },
  getCase(caseId) { return this.request(`/api/cases/${caseId}/`); },
  createCase(data) {
    return this.request('/api/cases/', {
      method: 'POST',
      body: JSON.stringify(data)
    });
  },
  addCaseNote(caseId, note) {
    return this.request(`/api/cases/${caseId}/notes/`, {
      method: 'POST',
      body: JSON.stringify({ note })
    });
  },
  resolveCase(caseId, resolution, notes) {
    return this.request(`/api/cases/${caseId}/resolve/`, {
      method: 'PATCH',
      body: JSON.stringify({ resolution, notes })
    });
  }
};

// UI Feedback Toast
function showToast(message, type = 'success') {
  let container = document.getElementById('toast-container');
  if (!container) {
    container = document.createElement('div');
    container.id = 'toast-container';
    container.className = 'toast-container';
    document.body.appendChild(container);
  }

  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.setAttribute('role', 'alert');
  toast.innerHTML = `<span>${message}</span>`;
  container.appendChild(toast);

  // Trigger enter animation
  requestAnimationFrame(() => toast.classList.add('show'));

  setTimeout(() => {
    toast.classList.remove('show');
    setTimeout(() => toast.remove(), 250);
  }, 4000);
}

// Modal Handlers with Keyboard Accessibility
function openModal(modalId) {
  const modal = document.getElementById(modalId);
  if (modal) {
    modal.classList.add('open');
    document.body.style.overflow = 'hidden';
    const firstInput = modal.querySelector('input, select, textarea, button:not(.modal-close)');
    if (firstInput) firstInput.focus();
  }
}

function closeModal(modalId) {
  const modal = document.getElementById(modalId);
  if (modal) {
    modal.classList.remove('open');
    document.body.style.overflow = '';
  }
}

// Sample Transaction Loaders for Quick Interactive Testing
function loadSampleTransaction(scenario) {
  const form = document.getElementById('singleTxForm');
  if (!form) return;

  if (scenario === 'suspicious_transfer') {
    // Drained origin transfer: highly fraudulent signature
    if (form.step) form.step.value = '1';
    if (form.type) form.type.value = 'TRANSFER';
    if (form.amount) form.amount.value = '450000.00';
    if (form.oldbalanceOrg) form.oldbalanceOrg.value = '450000.00';
    if (form.newbalanceOrig) form.newbalanceOrig.value = '0.00';
    if (form.oldbalanceDest) form.oldbalanceDest.value = '0.00';
    if (form.newbalanceDest) form.newbalanceDest.value = '0.00';
    if (form.nameOrig) form.nameOrig.value = 'C_HIGH_RISK_ORIGIN';
    if (form.nameDest) form.nameDest.value = 'M_DESTINATION_ACCOUNT';
    showToast('Loaded high-risk drained transfer scenario', 'warning');
  } else if (scenario === 'benign_payment') {
    // Normal small payment
    if (form.step) form.step.value = '2';
    if (form.type) form.type.value = 'PAYMENT';
    if (form.amount) form.amount.value = '124.50';
    if (form.oldbalanceOrg) form.oldbalanceOrg.value = '1500.00';
    if (form.newbalanceOrig) form.newbalanceOrig.value = '1375.50';
    if (form.oldbalanceDest) form.oldbalanceDest.value = '500.00';
    if (form.newbalanceDest) form.newbalanceDest.value = '624.50';
    if (form.nameOrig) form.nameOrig.value = 'C_MERCHANT_CLIENT';
    if (form.nameDest) form.nameDest.value = 'M_RETAIL_OUTLET';
    showToast('Loaded benign retail payment scenario', 'success');
  } else if (scenario === 'cash_out_spike') {
    // Suspicious cash out
    if (form.step) form.step.value = '3';
    if (form.type) form.type.value = 'CASH_OUT';
    if (form.amount) form.amount.value = '185000.00';
    if (form.oldbalanceOrg) form.oldbalanceOrg.value = '185000.00';
    if (form.newbalanceOrig) form.newbalanceOrig.value = '0.00';
    if (form.oldbalanceDest) form.oldbalanceDest.value = '25000.00';
    if (form.newbalanceDest) form.newbalanceDest.value = '210000.00';
    if (form.nameOrig) form.nameOrig.value = 'C_CASHOUT_VICTIM';
    if (form.nameDest) form.nameDest.value = 'M_ATM_AGENT';
    showToast('Loaded cash-out liquidation scenario', 'warning');
  }
}

// Inline Alert Transition with Instant Feedback
async function handleInlineAlertTransition(alertId, newStatus, btnElement) {
  try {
    btnElement.disabled = true;
    btnElement.textContent = 'Updating...';
    await api.transitionAlert(alertId, newStatus);
    showToast(`Alert ${alertId} transitioned to ${newStatus}`, 'success');
    
    // Find the row and update badge or remove
    const row = btnElement.closest('tr');
    if (row) {
      if (newStatus === 'DISMISSED' || newStatus === 'RESOLVED') {
        row.style.opacity = '0.4';
        setTimeout(() => row.remove(), 400);
      } else {
        const statusBadge = row.querySelector('.status-badge');
        if (statusBadge) statusBadge.textContent = newStatus;
        btnElement.remove();
      }
    }
  } catch (err) {
    showToast(err.message || 'Failed to update alert', 'error');
    btnElement.disabled = false;
    btnElement.textContent = 'Retry';
  }
}

// Initialize Global Handlers
document.addEventListener('DOMContentLoaded', () => {
  // Mobile Sidebar Toggle
  const toggleBtn = document.getElementById('mobile-sidebar-toggle');
  const sidebar = document.querySelector('.app-sidebar');

  if (toggleBtn && sidebar) {
    toggleBtn.addEventListener('click', () => {
      sidebar.classList.toggle('mobile-open');
    });
  }

  // Close modals on outside backdrop click or Escape key
  document.querySelectorAll('.modal-backdrop').forEach(b => {
    b.addEventListener('click', (e) => {
      if (e.target === b) {
        b.classList.remove('open');
        document.body.style.overflow = '';
      }
    });
  });

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      document.querySelectorAll('.modal-backdrop.open').forEach(b => {
        b.classList.remove('open');
        document.body.style.overflow = '';
      });
    }
  });
});
