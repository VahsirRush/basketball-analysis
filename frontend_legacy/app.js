// DOM Elements
const dropZone = document.getElementById('dropZone');
const fileInput = document.createElement('input');
fileInput.type = 'file';
fileInput.accept = 'video/mp4,video/quicktime,video/x-msvideo';

// State
let currentVideo = null;
let analysisResults = null;
let currentFilter = 'all';

// Restore analysisResults from localStorage if available
if (localStorage.getItem('analysisResults')) {
    try {
        analysisResults = JSON.parse(localStorage.getItem('analysisResults'));
        console.log('Restored analysisResults from localStorage:', analysisResults);
    } catch (e) {
        analysisResults = null;
    }
}

// Event Listeners
document.addEventListener('DOMContentLoaded', () => {
    initializeEventListeners();
});

function initializeEventListeners() {
    console.log('initializeEventListeners called');
    // File Upload
    if (dropZone) {
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });
        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('dragover');
        });
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            const files = e.dataTransfer.files;
            handleFiles(files);
        });
        dropZone.addEventListener('click', () => {
            fileInput.click();
        });
    }
    fileInput.addEventListener('change', (e) => {
        handleFiles(e.target.files);
    });
    // Import Video Button
    const importVideoBtn = document.querySelector('.btn-secondary');
    if (importVideoBtn) {
        importVideoBtn.addEventListener('click', () => {
            fileInput.click();
        });
    }
    // Start Analysis Button
    const startAnalysisBtn = document.querySelector('.btn-primary');
    if (startAnalysisBtn) {
        startAnalysisBtn.addEventListener('click', startAnalysis);
    }
    // Filter Select
    const filterSelect = document.querySelector('.filter-select');
    if (filterSelect) {
        filterSelect.addEventListener('change', (e) => {
            currentFilter = e.target.value;
            filterResults();
        });
    }
    // Search Button
    const searchBtn = document.querySelector('.filters .btn-icon');
    if (searchBtn) {
        searchBtn.addEventListener('click', () => {
            const searchTerm = prompt('Enter search term:');
            if (searchTerm) {
                searchResults(searchTerm);
            }
        });
    }
    // Analysis Panel
    const btnPlays = document.querySelectorAll('.btn-play');
    console.log('Found', btnPlays.length, '.btn-play elements');
    btnPlays.forEach(btn => {
        btn.addEventListener('click', (e) => {
            const playCard = e.target.closest('.play-card');
            if (!playCard) {
                showNotification('Play card not found.', 'error');
                return;
            }
            showAnalysisPanel(playCard);
        });
    });
    const closePanelBtn = document.querySelector('.analysis-panel .btn-icon');
    if (closePanelBtn) {
        closePanelBtn.addEventListener('click', () => {
            hideAnalysisPanel();
        });
    }
    // Sidebar Navigation
    document.querySelectorAll('.nav-item').forEach((item, idx) => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            let section = '';
            switch(idx) {
                case 0: section = 'Video Analysis'; break;
                case 1: section = 'Play Statistics'; break;
                case 2: section = 'Play Library'; break;
                case 3: section = 'Settings'; break;
                default: section = 'Video Analysis';
            }
            handleNavigation(section);
        });
    });
}

function handleNavigation(section) {
    console.log('Navigating to section:', section);
    document.querySelectorAll('.nav-item').forEach(item => {
        item.classList.remove('active');
    });
    const navItems = document.querySelectorAll('.nav-item');
    switch(section) {
        case 'Video Analysis': navItems[0]?.classList.add('active'); break;
        case 'Play Statistics': navItems[1]?.classList.add('active'); break;
        case 'Play Library': navItems[2]?.classList.add('active'); break;
        case 'Settings': navItems[3]?.classList.add('active'); break;
    }
    const mainContent = document.querySelector('.main-content');
    switch(section) {
        case 'Video Analysis':
            console.log('Rendering Video Analysis section');
            mainContent.innerHTML = `
                <header class="top-bar">
                    <h1>Video Analysis</h1>
                    <div class="actions">
                        <button class="btn-secondary">
                            <span class="icon">📁</span>
                            Import Video
                        </button>
                        <button class="btn-primary">
                            <span class="icon">▶️</span>
                            Start Analysis
                        </button>
                    </div>
                </header>
                <section class="upload-section">
                    <div class="upload-container">
                        <div class="upload-area" id="dropZone">
                            <img src="assets/upload-icon.svg" alt="Upload">
                            <h3>Drag and drop your video here</h3>
                            <p>or</p>
                            <button class="btn-secondary">Browse Files</button>
                            <p class="file-types">Supported formats: MP4, MOV, AVI</p>
                        </div>
                    </div>
                </section>
                <section class="results-section">
                    <div class="results-grid"></div>
                </section>
            `;
            if (!analysisResults && localStorage.getItem('analysisResults')) {
                try {
                    analysisResults = JSON.parse(localStorage.getItem('analysisResults'));
                    console.log('Restored analysisResults from localStorage (navigation):', analysisResults);
                } catch (e) {
                    analysisResults = null;
                }
            }
            if (analysisResults) {
                console.log('Updating results grid with existing analysisResults');
                updateResultsGrid(analysisResults);
            }
            initializeEventListeners();
            break;
        case 'Play Statistics':
            mainContent.innerHTML = `
                <header class="top-bar">
                    <h1>Play Statistics</h1>
                </header>
                <section class="stats-section">
                    <div class="stats-grid">
                        <div class="stat-card">
                            <h3>Total Plays</h3>
                            <p class="stat-value">0</p>
                        </div>
                        <div class="stat-card">
                            <h3>Success Rate</h3>
                            <p class="stat-value">0%</p>
                        </div>
                        <div class="stat-card">
                            <h3>Avg. Duration</h3>
                            <p class="stat-value">0s</p>
                        </div>
                    </div>
                </section>
            `;
            initializeEventListeners();
            break;
        case 'Play Library':
            mainContent.innerHTML = `
                <header class="top-bar">
                    <h1>Play Library</h1>
                    <div class="actions">
                        <button class="btn-secondary">
                            <span class="icon">📁</span>
                            Import Play
                        </button>
                    </div>
                </header>
                <section class="library-section">
                    <div class="library-grid">
                        <!-- Library content will be dynamically updated -->
                    </div>
                </section>
            `;
            initializeEventListeners();
            break;
        case 'Settings':
            mainContent.innerHTML = `
                <header class="top-bar">
                    <h1>Settings</h1>
                </header>
                <section class="settings-section">
                    <div class="settings-card">
                        <h3>General Settings</h3>
                        <div class="setting-item">
                            <label>Theme</label>
                            <select>
                                <option>Light</option>
                                <option>Dark</option>
                            </select>
                        </div>
                        <div class="setting-item">
                            <label>Language</label>
                            <select>
                                <option>English</option>
                                <option>Spanish</option>
                            </select>
                        </div>
                    </div>
                </section>
            `;
            initializeEventListeners();
            break;
    }
}

function filterResults() {
    if (!analysisResults) return;
    
    const filteredPlays = currentFilter === 'all' 
        ? analysisResults.plays 
        : analysisResults.plays.filter(play => play.type.toLowerCase().includes(currentFilter.toLowerCase()));
    
    updateResultsGrid({ plays: filteredPlays });
}

function searchResults(term) {
    if (!analysisResults) return;
    
    const searchResults = analysisResults.plays.filter(play => 
        play.type.toLowerCase().includes(term.toLowerCase()) ||
        play.players.some(player => player.number.toString().includes(term))
    );
    
    updateResultsGrid({ plays: searchResults });
}

function handleFiles(files) {
    if (files.length === 0) return;

    const file = files[0];
    if (!isValidVideoFile(file)) {
        showNotification('Please upload a valid video file (MP4, MOV, or AVI)', 'error');
        return;
    }

    currentVideo = file;
    showNotification('Video uploaded successfully', 'success');
    updateUploadArea(file);
}

function isValidVideoFile(file) {
    const validTypes = ['video/mp4', 'video/quicktime', 'video/x-msvideo'];
    return validTypes.includes(file.type);
}

function updateUploadArea(file) {
    const uploadArea = document.querySelector('.upload-area');
    uploadArea.innerHTML = `
        <div class="upload-preview">
            <video src="${URL.createObjectURL(file)}" controls></video>
            <div class="upload-info">
                <h3>${file.name}</h3>
                <p>${formatFileSize(file.size)}</p>
            </div>
        </div>
    `;
}

async function startAnalysis() {
    if (!currentVideo) {
        showNotification('Please upload a video first', 'error');
        return;
    }

    showLoadingState();
    try {
        console.log('Starting analysis with video:', currentVideo.name);
        const formData = new FormData();
        formData.append('video', currentVideo);

        console.log('Sending request to API...');
        const response = await fetch('http://localhost:5001/api/analyze', {
            method: 'POST',
            body: formData
        });

        console.log('Response status:', response.status);
        if (!response.ok) {
            const errorData = await response.json();
            console.error('API Error:', errorData);
            throw new Error(errorData.error || 'Analysis failed');
        }

        const data = await response.json();
        console.log('Received analysis results:', data);
        analysisResults = data;
        localStorage.setItem('analysisResults', JSON.stringify(analysisResults));
        updateResultsGrid(analysisResults);
        showNotification('Analysis completed successfully', 'success');
    } catch (error) {
        console.error('Analysis error:', error);
        showNotification('Analysis failed: ' + error.message, 'error');
    } finally {
        hideLoadingState();
    }
}

function updateResultsGrid(results) {
    console.log('Calling updateResultsGrid with results:', results);
    let grid = document.querySelector('.results-grid');
    if (!grid) {
        console.log('No .results-grid found, creating one.');
        const resultsSection = document.querySelector('.results-section');
        grid = document.createElement('div');
        grid.className = 'results-grid';
        resultsSection.appendChild(grid);
    }
    grid.innerHTML = results.plays.map(play => createPlayCard(play)).join('');
    console.log('Updated .results-grid innerHTML:', grid.innerHTML);
    initializeEventListeners();
}

function createPlayCard(play) {
    return `
        <div class="play-card" data-play-id="${play.id}">
            <div class="play-video">
                <img src="${play.thumbnail}" alt="Play Preview">
                <div class="play-overlay">
                    <span class="play-duration">${formatDuration(play.duration)}</span>
                    <button class="btn-play">▶️</button>
                </div>
            </div>
            <div class="play-info">
                <h3>${play.type}</h3>
                <div class="play-stats">
                    <span class="stat">
                        <span class="icon">⏱️</span>
                        ${formatDuration(play.duration)}
                    </span>
                    <span class="stat">
                        <span class="icon">👥</span>
                        ${play.playerCount} Players
                    </span>
                </div>
                <div class="play-actions">
                    <button class="btn-icon" onclick="showStats(${play.id})">📊</button>
                    <button class="btn-icon" onclick="downloadPlay(${play.id})">💾</button>
                    <button class="btn-icon" onclick="sharePlay(${play.id})">↗️</button>
                </div>
            </div>
        </div>
    `;
}

function generatePlayDiagram(play) {
    // Court dimensions (outer lines)
    const courtWidth = 540; // 600 - 2*30
    const courtHeight = 840; // 900 - 2*30
    const margin = 30;
    const svgNS = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(svgNS, 'svg');
    svg.setAttribute('width', '100%');
    svg.setAttribute('height', courtHeight);
    svg.setAttribute('viewBox', `0 0 ${courtWidth} ${courtHeight}`);
    svg.style.display = 'block';
    svg.style.margin = '0 auto';
    svg.style.background = '#f7e6c4'; // Only for the court area
    svg.style.borderRadius = '16px';

    // Draw court (no extra margin)
    const courtRect = document.createElementNS(svgNS, 'rect');
    courtRect.setAttribute('x', 0);
    courtRect.setAttribute('y', 0);
    courtRect.setAttribute('width', courtWidth);
    courtRect.setAttribute('height', courtHeight);
    courtRect.setAttribute('rx', 16);
    courtRect.setAttribute('fill', '#f7e6c4');
    courtRect.setAttribute('stroke', 'none');
    svg.appendChild(courtRect);
    // Outer lines
    const outer = document.createElementNS(svgNS, 'rect');
    outer.setAttribute('x', 0);
    outer.setAttribute('y', 0);
    outer.setAttribute('width', courtWidth);
    outer.setAttribute('height', courtHeight);
    outer.setAttribute('rx', 0);
    outer.setAttribute('fill', 'none');
    outer.setAttribute('stroke', '#fff');
    outer.setAttribute('stroke-width', 6);
    svg.appendChild(outer);
    // Key (paint)
    const keyWidth = 120;
    const keyHeight = 240;
    const keyRect = document.createElementNS(svgNS, 'rect');
    keyRect.setAttribute('x', courtWidth/2 - keyWidth/2);
    keyRect.setAttribute('y', 90);
    keyRect.setAttribute('width', keyWidth);
    keyRect.setAttribute('height', keyHeight);
    keyRect.setAttribute('fill', 'none');
    keyRect.setAttribute('stroke', '#fff');
    keyRect.setAttribute('stroke-width', 6);
    svg.appendChild(keyRect);
    // Free throw circle
    const ftCircle = document.createElementNS(svgNS, 'ellipse');
    ftCircle.setAttribute('cx', courtWidth/2);
    ftCircle.setAttribute('cy', 90 + keyHeight/2);
    ftCircle.setAttribute('rx', 60);
    ftCircle.setAttribute('ry', 60);
    ftCircle.setAttribute('stroke', '#fff');
    ftCircle.setAttribute('stroke-width', 6);
    ftCircle.setAttribute('fill', 'none');
    svg.appendChild(ftCircle);
    // Rim
    const rim = document.createElementNS(svgNS, 'ellipse');
    rim.setAttribute('cx', courtWidth/2);
    rim.setAttribute('cy', 90);
    rim.setAttribute('rx', 18);
    rim.setAttribute('ry', 8);
    rim.setAttribute('stroke', '#fff');
    rim.setAttribute('stroke-width', 6);
    rim.setAttribute('fill', 'none');
    svg.appendChild(rim);
    // Backboard
    const backboard = document.createElementNS(svgNS, 'rect');
    backboard.setAttribute('x', courtWidth/2 - 30);
    backboard.setAttribute('y', 78);
    backboard.setAttribute('width', 60);
    backboard.setAttribute('height', 6);
    backboard.setAttribute('fill', '#fff');
    svg.appendChild(backboard);
    // Three-point arc
    const arc = document.createElementNS(svgNS, 'path');
    arc.setAttribute('d', `M 40 ${courtHeight} A 220 220 0 0 1 ${courtWidth - 40} ${courtHeight}`);
    arc.setAttribute('stroke', '#fff');
    arc.setAttribute('stroke-width', 6);
    arc.setAttribute('fill', 'none');
    svg.appendChild(arc);

    // Draw player movements (lines/arrows first)
    if (play.players && play.players.length > 0 && play.movements && play.movements.length > 0) {
        play.players.forEach(player => {
            const playerMovements = play.movements.filter(m => m.playerNumber === player.number);
            playerMovements.forEach(movement => {
                // Scale positions to fit within court
                const startX = (movement.start.x / 100) * courtWidth;
                const startY = (movement.start.y / 100) * courtHeight;
                const endX = (movement.end.x / 100) * courtWidth;
                const endY = (movement.end.y / 100) * courtHeight;
                const dx = endX - startX;
                const dy = endY - startY;
                const angle = Math.atan2(dy, dx);
                // Cut: solid black line with arrow
                if (movement.type === 'cut') {
                    const line = document.createElementNS(svgNS, 'line');
                    line.setAttribute('x1', startX);
                    line.setAttribute('y1', startY);
                    line.setAttribute('x2', endX);
                    line.setAttribute('y2', endY);
                    line.setAttribute('stroke', '#111');
                    line.setAttribute('stroke-width', 10);
                    line.setAttribute('stroke-linecap', 'round');
                    svg.appendChild(line);
                    // Arrowhead
                    const arrow = document.createElementNS(svgNS, 'polygon');
                    const arrowLength = 48;
                    const arrowWidth = 24;
                    const ax = endX - Math.cos(angle) * 18;
                    const ay = endY - Math.sin(angle) * 18;
                    const points = [
                        [ax, ay],
                        [ax - arrowLength * Math.cos(angle - 0.3), ay - arrowLength * Math.sin(angle - 0.3)],
                        [ax - arrowLength * Math.cos(angle + 0.3), ay - arrowLength * Math.sin(angle + 0.3)]
                    ];
                    arrow.setAttribute('points', points.map(p => p.join(',')).join(' '));
                    arrow.setAttribute('fill', '#111');
                    svg.appendChild(arrow);
                }
                // Screen: solid black line with T-end
                else if (movement.type === 'screen') {
                    const line = document.createElementNS(svgNS, 'line');
                    line.setAttribute('x1', startX);
                    line.setAttribute('y1', startY);
                    line.setAttribute('x2', endX);
                    line.setAttribute('y2', endY);
                    line.setAttribute('stroke', '#111');
                    line.setAttribute('stroke-width', 10);
                    line.setAttribute('stroke-linecap', 'round');
                    svg.appendChild(line);
                    // T-end
                    const tLen = 48;
                    const tAngle = angle + Math.PI / 2;
                    const tx1 = endX + Math.cos(tAngle) * tLen / 2;
                    const ty1 = endY + Math.sin(tAngle) * tLen / 2;
                    const tx2 = endX - Math.cos(tAngle) * tLen / 2;
                    const ty2 = endY - Math.sin(tAngle) * tLen / 2;
                    const tLine = document.createElementNS(svgNS, 'line');
                    tLine.setAttribute('x1', tx1);
                    tLine.setAttribute('y1', ty1);
                    tLine.setAttribute('x2', tx2);
                    tLine.setAttribute('y2', ty2);
                    tLine.setAttribute('stroke', '#111');
                    tLine.setAttribute('stroke-width', 10);
                    tLine.setAttribute('stroke-linecap', 'round');
                    svg.appendChild(tLine);
                }
                // Pass: dashed black line with arrow
                else if (movement.type === 'pass') {
                    const line = document.createElementNS(svgNS, 'line');
                    line.setAttribute('x1', startX);
                    line.setAttribute('y1', startY);
                    line.setAttribute('x2', endX);
                    line.setAttribute('y2', endY);
                    line.setAttribute('stroke', '#111');
                    line.setAttribute('stroke-width', 10);
                    line.setAttribute('stroke-dasharray', '42,28');
                    line.setAttribute('stroke-linecap', 'round');
                    svg.appendChild(line);
                    // Arrowhead
                    const arrow = document.createElementNS(svgNS, 'polygon');
                    const arrowLength = 48;
                    const arrowWidth = 24;
                    const ax = endX - Math.cos(angle) * 18;
                    const ay = endY - Math.sin(angle) * 18;
                    const points = [
                        [ax, ay],
                        [ax - arrowLength * Math.cos(angle - 0.3), ay - arrowLength * Math.sin(angle - 0.3)],
                        [ax - arrowLength * Math.cos(angle + 0.3), ay - arrowLength * Math.sin(angle + 0.3)]
                    ];
                    arrow.setAttribute('points', points.map(p => p.join(',')).join(' '));
                    arrow.setAttribute('fill', '#111');
                    svg.appendChild(arrow);
                }
                // Dribble: wavy black line
                else if (movement.type === 'dribble') {
                    const dribblePath = document.createElementNS(svgNS, 'path');
                    const steps = 40;
                    let d = `M ${startX} ${startY}`;
                    for (let i = 1; i <= steps; i++) {
                        const t = i / steps;
                        const x = startX + dx * t;
                        const y = startY + dy * t + Math.sin(t * Math.PI * 6) * 28;
                        d += ` L ${x} ${y}`;
                    }
                    dribblePath.setAttribute('d', d);
                    dribblePath.setAttribute('stroke', '#111');
                    dribblePath.setAttribute('stroke-width', 10);
                    dribblePath.setAttribute('fill', 'none');
                    dribblePath.setAttribute('stroke-linecap', 'round');
                    svg.appendChild(dribblePath);
                    // Arrowhead at end
                    const angle2 = Math.atan2(dy, dx);
                    const arrow = document.createElementNS(svgNS, 'polygon');
                    const arrowLength = 48;
                    const arrowWidth = 24;
                    const ax = endX - Math.cos(angle2) * 18;
                    const ay = endY - Math.sin(angle2) * 18;
                    const points = [
                        [ax, ay],
                        [ax - arrowLength * Math.cos(angle2 - 0.3), ay - arrowLength * Math.sin(angle2 - 0.3)],
                        [ax - arrowLength * Math.cos(angle2 + 0.3), ay - arrowLength * Math.sin(angle2 + 0.3)]
                    ];
                    arrow.setAttribute('points', points.map(p => p.join(',')).join(' '));
                    arrow.setAttribute('fill', '#111');
                    svg.appendChild(arrow);
                }
                // Handoff: line with two crossing lines at the handoff point
                else if (movement.type === 'handoff') {
                    const line = document.createElementNS(svgNS, 'line');
                    line.setAttribute('x1', startX);
                    line.setAttribute('y1', startY);
                    line.setAttribute('x2', endX);
                    line.setAttribute('y2', endY);
                    line.setAttribute('stroke', '#111');
                    line.setAttribute('stroke-width', 10);
                    line.setAttribute('stroke-linecap', 'round');
                    svg.appendChild(line);
                    // Two crossing lines at the handoff point (end)
                    const crossLen = 42;
                    for (let i = -1; i <= 1; i += 2) {
                        const cross = document.createElementNS(svgNS, 'line');
                        const crossAngle = angle + i * Math.PI / 4;
                        const cx1 = endX - Math.cos(crossAngle) * crossLen / 2;
                        const cy1 = endY - Math.sin(crossAngle) * crossLen / 2;
                        const cx2 = endX + Math.cos(crossAngle) * crossLen / 2;
                        const cy2 = endY + Math.sin(crossAngle) * crossLen / 2;
                        cross.setAttribute('x1', cx1);
                        cross.setAttribute('y1', cy1);
                        cross.setAttribute('x2', cx2);
                        cross.setAttribute('y2', cy2);
                        cross.setAttribute('stroke', '#111');
                        cross.setAttribute('stroke-width', 10);
                        svg.appendChild(cross);
                    }
                }
                // (Optional) Action label in red, offset further
                if (movement.type && ['cut','screen','pass','dribble','handoff'].includes(movement.type)) {
                    const label = document.createElementNS(svgNS, 'text');
                    label.setAttribute('x', endX + 36);
                    label.setAttribute('y', endY - 36);
                    label.setAttribute('font-size', '2.4rem');
                    label.setAttribute('font-family', 'Inter, sans-serif');
                    label.setAttribute('font-weight', 'bold');
                    label.setAttribute('fill', '#e11d48');
                    label.textContent = movement.type.charAt(0).toUpperCase() + movement.type.slice(1);
                    svg.appendChild(label);
                }
            });
        });
    }

    // Draw players (large, bold, black numbers in circles)
    if (play.players && play.players.length > 0) {
        play.players.forEach(player => {
            // Scale positions to fit within court
            const px = (player.position.x / 100) * courtWidth;
            const py = (player.position.y / 100) * courtHeight;
            // Draw circle
            const circle = document.createElementNS(svgNS, 'circle');
            circle.setAttribute('cx', px);
            circle.setAttribute('cy', py);
            circle.setAttribute('r', 32);
            circle.setAttribute('stroke', '#111');
            circle.setAttribute('stroke-width', 6);
            circle.setAttribute('fill', '#f7e6c4');
            svg.appendChild(circle);
            // Draw number
            const text = document.createElementNS(svgNS, 'text');
            text.setAttribute('x', px);
            text.setAttribute('y', py + 16);
            text.setAttribute('text-anchor', 'middle');
            text.setAttribute('font-size', '2.8rem');
            text.setAttribute('font-family', 'Inter, sans-serif');
            text.setAttribute('font-weight', '900');
            text.setAttribute('fill', '#111');
            text.textContent = player.number;
            svg.appendChild(text);
        });
    }

    // Wrap SVG in a div for layout
    const wrapper = document.createElement('div');
    wrapper.className = 'play-diagram';
    wrapper.appendChild(svg);
    return wrapper;
}

function showAnalysisPanel(playCard) {
    if (!analysisResults || !Array.isArray(analysisResults.plays)) {
        showNotification('No analysis results available. Please run analysis first.', 'error');
        return;
    }
    const playId = playCard.dataset.playId;
    const play = analysisResults.plays.find(p => p.id === playId || p.id === Number(playId));
    console.log('showAnalysisPanel called for playId:', playId, 'play:', play);
    if (!play) {
        showNotification('Play data not found.', 'error');
        return;
    }
    play.offensiveEnd = determineOffensiveEnd(play);
    const panel = document.querySelector('.analysis-panel');
    if (!panel) {
        showNotification('Analysis panel not found.', 'error');
        return;
    }
    panel.innerHTML = `
        <div class="panel-header">
            <h2>Play Analysis</h2>
            <button class="btn-icon">✕</button>
        </div>
        <div class="panel-content">
            <div class="analysis-section">
                <h3>Play Diagram</h3>
                <div class="play-diagram-container">
                    ${generatePlayDiagram(play).outerHTML}
                </div>
            </div>
            <div class="analysis-section">
                <h3>Player Movements</h3>
                <div class="movement-list">
                    ${play.movements.map(movement => `
                        <div class="movement-item">
                            <span class="player">#${movement.playerNumber}</span>
                            <span class="movement">${movement.type}</span>
                            <span class="confidence">${(movement.confidence * 100).toFixed(0)}%</span>
                        </div>
                    `).join('')}
                </div>
            </div>
        </div>
    `;
    panel.classList.add('active');
    const newDiagram = panel.querySelector('.play-diagram');
    if (newDiagram) {
        const playBtn = newDiagram.querySelector('.play-control-btn');
        const resetBtn = newDiagram.querySelector('.play-control-btn:last-child');
        if (playBtn) playBtn.onclick = () => animatePlay(newDiagram);
        if (resetBtn) resetBtn.onclick = () => resetPlay(newDiagram);
    }
}

function hideAnalysisPanel() {
    const panel = document.querySelector('.analysis-panel');
    if (panel) {
        panel.classList.remove('active');
    }
}

function determineOffensiveEnd(play) {
    const topPlayers = play.players.filter(p => p.position.y < 50).length;
    const bottomPlayers = play.players.filter(p => p.position.y >= 50).length;
    
    return topPlayers > bottomPlayers ? 'top' : 'bottom';
}

function formatFileSize(bytes) {
    const units = ['B', 'KB', 'MB', 'GB'];
    let size = bytes;
    let unitIndex = 0;

    while (size >= 1024 && unitIndex < units.length - 1) {
        size /= 1024;
        unitIndex++;
    }

    return `${size.toFixed(1)} ${units[unitIndex]}`;
}

function formatDuration(seconds) {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
}

function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.textContent = message;

    document.body.appendChild(notification);

    setTimeout(() => {
        notification.classList.add('show');
    }, 100);

    setTimeout(() => {
        notification.classList.remove('show');
        setTimeout(() => {
            notification.remove();
        }, 300);
    }, 3000);
}

function showLoadingState() {
    const loading = document.createElement('div');
    loading.className = 'loading-overlay';
    loading.innerHTML = `
        <div class="loading-spinner"></div>
        <p>Analyzing video...</p>
    `;
    document.body.appendChild(loading);
}

function hideLoadingState() {
    const loading = document.querySelector('.loading-overlay');
    if (loading) {
        loading.remove();
    }
}

function showStats(playId) {
    console.log('Show stats for play:', playId);
}

function downloadPlay(playId) {
    console.log('Download play:', playId);
}

function sharePlay(playId) {
    console.log('Share play:', playId);
} 