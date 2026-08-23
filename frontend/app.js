document.addEventListener('DOMContentLoaded', () => {
    // ---- Navigation ----
    const navBtns = document.querySelectorAll('.nav-btn');
    const views = document.querySelectorAll('.view');
    
    navBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const viewId = `view-${btn.dataset.view}`;
            
            navBtns.forEach(b => b.classList.remove('active'));
            views.forEach(v => v.classList.add('hidden'));
            
            btn.classList.add('active');
            document.getElementById(viewId).classList.remove('hidden');
            document.getElementById(viewId).classList.add('active');
            
            if (btn.dataset.view === 'history') {
                loadHistory();
            }
        });
    });

    // ---- Health Check ----
    async function checkHealth() {
        try {
            const res = await fetch('/api/health');
            const data = await res.json();
            const indicator = document.querySelector('.status-indicator');
            const text = document.querySelector('.status-text');
            
            if (data.status === 'ok' && data.model_loaded) {
                indicator.className = 'status-indicator ok';
                text.textContent = 'Model loaded (CPU)';
            } else {
                indicator.className = 'status-indicator error';
                text.textContent = 'Model error';
            }
        } catch (err) {
            console.error(err);
            document.querySelector('.status-indicator').className = 'status-indicator error';
            document.querySelector('.status-text').textContent = 'Offline';
        }
    }
    
    // Poll health initially
    checkHealth();

    // ---- Single Inference ----
    const fileInput = document.getElementById('file-input');
    const uploadPrompt = document.getElementById('upload-prompt');
    const uploadPreview = document.getElementById('upload-preview');
    const previewImg = document.getElementById('preview-img');
    const previewFilename = document.getElementById('preview-filename');
    const btnRemove = document.getElementById('btn-remove');
    const btnAnalyze = document.getElementById('btn-analyze');
    const uploadZone = document.getElementById('upload-zone');
    
    let currentFile = null;

    uploadPrompt.addEventListener('click', () => fileInput.click());
    
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('dragover');
    });
    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('dragover');
    });
    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            handleFileSelection(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFileSelection(e.target.files[0]);
        }
    });

    function handleFileSelection(file) {
        currentFile = file;
        previewFilename.textContent = file.name;
        
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImg.src = e.target.result;
            uploadPrompt.classList.add('hidden');
            uploadPreview.classList.remove('hidden');
            btnAnalyze.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    btnRemove.addEventListener('click', () => {
        currentFile = null;
        fileInput.value = '';
        uploadPreview.classList.add('hidden');
        uploadPrompt.classList.remove('hidden');
        btnAnalyze.disabled = true;
    });
    
    document.getElementById('btn-new').addEventListener('click', () => {
        document.getElementById('results-container').classList.add('hidden');
        uploadZone.classList.remove('hidden');
        document.querySelector('.actions').classList.remove('hidden');
        btnRemove.click();
    });

    btnAnalyze.addEventListener('click', async () => {
        if (!currentFile) return;
        
        uploadZone.classList.add('hidden');
        document.querySelector('.actions').classList.add('hidden');
        document.getElementById('loading-state').classList.remove('hidden');
        
        const formData = new FormData();
        formData.append('file', currentFile);
        
        try {
            const res = await fetch('/api/infer', {
                method: 'POST',
                body: formData
            });
            
            const data = await res.json();
            document.getElementById('loading-state').classList.add('hidden');
            
            if (res.ok && data.status === 'success') {
                showResults(data);
            } else {
                alert('Error: ' + (data.detail || data.error || 'Unknown error'));
                document.getElementById('btn-new').click();
            }
        } catch (err) {
            alert('Request failed: ' + err.message);
            document.getElementById('loading-state').classList.add('hidden');
            document.getElementById('btn-new').click();
        }
    });
    
    function showResults(data) {
        document.getElementById('results-container').classList.remove('hidden');
        document.getElementById('res-time').textContent = data.inference_time_seconds + 's';
        document.getElementById('res-findings').textContent = data.findings || 'No findings generated.';
        document.getElementById('res-impression').textContent = data.impression || 'No impression generated.';
    }

    // ---- Batch Inference ----
    const batchInput = document.getElementById('batch-file-input');
    const batchList = document.getElementById('batch-list');
    const btnBatchAnalyze = document.getElementById('btn-batch-analyze');
    let batchFiles = [];
    
    batchInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            batchFiles = Array.from(e.target.files);
            renderBatchList();
            btnBatchAnalyze.disabled = false;
        }
    });
    
    function renderBatchList() {
        batchList.innerHTML = `<p>${batchFiles.length} files selected.</p>`;
    }
    
    btnBatchAnalyze.addEventListener('click', async () => {
        if (!batchFiles.length) return;
        
        document.getElementById('batch-upload-zone').classList.add('hidden');
        batchList.classList.add('hidden');
        btnBatchAnalyze.classList.add('hidden');
        
        const loading = document.getElementById('batch-loading-state');
        loading.classList.remove('hidden');
        
        const formData = new FormData();
        batchFiles.forEach(f => formData.append('files', f));
        
        try {
            const res = await fetch('/api/infer/batch', {
                method: 'POST',
                body: formData
            });
            const data = await res.json();
            
            loading.classList.add('hidden');
            document.getElementById('batch-results').classList.remove('hidden');
            
            const tbody = document.getElementById('batch-results-body');
            tbody.innerHTML = '';
            
            data.results.forEach(r => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td>${r.image || r.image_path || 'Unknown'}</td>
                    <td><span class="badge" style="background:${r.status==='success'?'#dcfce7':'#fee2e2'};color:${r.status==='success'?'#166534':'#991b1b'}">${r.status}</span></td>
                    <td>${r.inference_time_seconds || '-'}s</td>
                    <td>
                        <button class="btn btn-secondary" onclick="alert('Findings:\\n${(r.findings||'').replace(/'/g, "\\'")}\\n\\nImpression:\\n${(r.impression||'').replace(/'/g, "\\'")}')">View</button>
                    </td>
                `;
                tbody.appendChild(tr);
            });
        } catch (err) {
            alert('Batch failed: ' + err.message);
            loading.classList.add('hidden');
        }
    });

    // ---- History ----
    const modal = document.getElementById('history-modal');
    document.getElementById('modal-close').addEventListener('click', () => {
        modal.classList.remove('show');
    });
    
    async function loadHistory() {
        try {
            const res = await fetch('/api/history');
            const items = await res.json();
            
            const tbody = document.getElementById('history-body');
            tbody.innerHTML = '';
            
            if (items.length === 0) {
                tbody.innerHTML = '<tr><td colspan="6" style="text-align: center;">No history found.</td></tr>';
                return;
            }
            
            items.forEach(item => {
                const date = new Date(item.timestamp).toLocaleString();
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td>${item.id}</td>
                    <td>${date}</td>
                    <td>${item.image_name}</td>
                    <td>${item.inference_time_seconds}</td>
                    <td>${item.status}</td>
                    <td>
                        <button class="btn btn-secondary btn-sm" onclick="viewHistoryItem(${item.id})">View</button>
                        <a href="/api/report/${item.id}/json" class="btn btn-outline btn-sm">JSON</a>
                    </td>
                `;
                tbody.appendChild(tr);
            });
        } catch (err) {
            console.error(err);
        }
    }
    
    window.viewHistoryItem = async function(id) {
        try {
            const res = await fetch('/api/report/' + id + '/json');
            const data = await res.json();
            
            const html = `
                <div style="margin-bottom: 20px;">
                    <strong>Image:</strong> ${data.image}<br>
                    <strong>Date:</strong> ${new Date(data.timestamp).toLocaleString()}<br>
                    <strong>Time:</strong> ${data.inference_time_seconds}s
                </div>
                <h3>Findings</h3>
                <div class="report-box">${data.findings}</div>
                <h3 style="margin-top:15px;">Impression</h3>
                <div class="report-box">${data.impression}</div>
                <div style="margin-top: 20px; text-align:right;">
                    <a href="/api/report/${id}/txt" class="btn btn-primary">Download TXT</a>
                </div>
            `;
            document.getElementById('modal-body').innerHTML = html;
            modal.classList.add('show');
        } catch (err) {
            alert('Failed to load item: ' + err.message);
        }
    }
});
