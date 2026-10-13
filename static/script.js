let jsmeApplet = null;

// Initialize JSME Editor
function jsmeOnLoad() {
    jsmeApplet = new JSApplet.JSME("jsme_container", "100%", "100%", {
        "options" : "oldlook,star,multipart"
    });
}

function goTo(pageId) {
    document.querySelectorAll('.page').forEach(page => {
        page.classList.add('hidden');
        page.classList.remove('active');
    });
    
    const target = document.getElementById(pageId);
    if (target) {
        target.classList.remove('hidden');
        // Small delay to allow CSS transition
        setTimeout(() => target.classList.add('active'), 10);
    }
}

function getSmilesFromEditor() {
    if (jsmeApplet) {
        const smiles = (jsmeApplet.smiles() || "").trim();
        if (!smiles) {
            alert("Please draw a chemical structure in the editor first.");
            return;
        }
        document.getElementById('mol-input').value = smiles;
        predictOdor();
    }
}

function updateSlider(id) {
    const val = document.getElementById('slider-' + id).value;
    document.getElementById('val-' + id).innerText = parseFloat(val).toFixed(1);
}

async function predictOdor() {
    const input = document.getElementById('mol-input').value.trim();
    if (!input) {
        alert("Please enter a molecule name or SMILES string.");
        return;
    }

    const resultsBox = document.getElementById('predict-results');
    const list = document.getElementById('predict-list');
    
    resultsBox.classList.remove('hidden');
    list.innerHTML = '<li>Analyzing molecule and predicting odor profile...</li>';

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ molecule: input })
        });

        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.detail || "Prediction failed");
        }

        list.innerHTML = ''; // Clear loading text
        
        // Remove legacy preview container if present
        const oldPreview = resultsBox.querySelector('.predict-img-container');
        if (oldPreview) oldPreview.remove();

        // High-definition Molecule Info Banner
        let banner = resultsBox.querySelector('.molecule-info-banner');
        if (!banner) {
            banner = document.createElement('div');
            banner.className = 'molecule-info-banner';
            resultsBox.insertBefore(banner, list);
        }
        
        const formulaText = data.formula ? data.formula : 'N/A';
        const weightText = data.mol_weight ? `${data.mol_weight} g/mol` : 'N/A';
        const isNovel = data.name.startsWith("Novel Molecule");
        
        banner.innerHTML = `
            <img class="mol-structure-img" src="/api/molecule_image?smiles=${encodeURIComponent(data.smiles)}" alt="${data.name}" />
            <div class="mol-meta-grid">
                <div class="mol-meta-item">
                    <span class="label">Molecule Name</span>
                    <span class="val name-tag">${data.name}</span>
                </div>
                <div class="mol-meta-item">
                    <span class="label">SMILES String</span>
                    <span class="val" title="${data.smiles}">${data.smiles}</span>
                </div>
                <div class="mol-meta-item">
                    <span class="label">Formula & Mass</span>
                    <span class="val">${formulaText} • ${weightText}</span>
                </div>
            </div>
        `;
        
        // Update Title
        const titleEl = resultsBox.querySelector('h3');
        if (titleEl) {
            if (!isNovel) {
                titleEl.innerHTML = `Predicted Odor Profile for <span style="color: #00f2fe;">${data.name}</span>:`;
            } else {
                titleEl.innerHTML = `Predicted Odor Profile for Novel Structure:`;
            }
        }

        // Populate list
        data.predictions.forEach(pred => {
            const li = document.createElement('li');
            const percentage = (pred.score * 100).toFixed(1);
            li.innerHTML = `
                <span><strong>${pred.odor}</strong> (${percentage}%)</span>
                <div class="score-bar-bg">
                    <div class="score-bar-fill" style="width: ${percentage}%"></div>
                </div>
            `;
            list.appendChild(li);
        });
        
    } catch (error) {
        list.innerHTML = `<li style="color: #ef4444;">Error: ${error.message}</li>`;
    }
}

async function generateMolecules() {
    // Collect target values
    const target = {};
    const ids = [
        'woody', 'fruity', 'floral', 'sweet', 'earthy',
        'citrus', 'herbal', 'spicy', 'green', 'musk',
        'vanilla', 'nutty', 'minty', 'fresh', 'balsamic',
        'roasted', 'waxy', 'sulfurous', 'rose', 'camphor'
    ];
    let sum = 0;
    
    ids.forEach(id => {
        const val = parseFloat(document.getElementById('slider-' + id).value);
        if (val > 0) {
            let labelName = id;
            if (labelName === 'camphor') labelName = 'camphoraceous';
            target[labelName] = val;
            sum += val;
        }
    });

    if (sum === 0) {
        alert("Please set at least one scent profile above 0.");
        return;
    }

    const resultsBox = document.getElementById('generate-results');
    const grid = document.getElementById('generate-grid');
    
    resultsBox.classList.remove('hidden');
    grid.innerHTML = '<p>Designing molecules...</p>';

    try {
        const response = await fetch('/api/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ target: target, n_samples: 1000 })
        });

        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.detail || "Generation failed");
        }

        grid.innerHTML = ''; // Clear loading text

        if (data.candidates.length === 0) {
            grid.innerHTML = '<p>No valid candidates found for this profile.</p>';
            return;
        }

        // Populate grid
        data.candidates.forEach(cand => {
            const card = document.createElement('div');
            card.className = 'mol-card';
            
            // Crisp RDKit 2D Vector Drawing (Always generated & centered)
            const img = document.createElement('img');
            img.src = `/api/molecule_image?smiles=${encodeURIComponent(cand.smiles)}`;
            img.alt = cand.name;
            img.loading = 'lazy';
            
            card.appendChild(img);
            
            const nameEl = document.createElement('h4');
            nameEl.style.margin = "5px 0";
            nameEl.innerText = cand.name;
            card.appendChild(nameEl);
            
            const p = document.createElement('p');
            p.style.fontSize = "0.8em";
            p.style.wordBreak = "break-all";
            p.innerText = cand.smiles;
            card.appendChild(p);
            
            const score = document.createElement('small');
            score.innerText = `Match Score: ${cand.score.toFixed(2)}`;
            card.appendChild(score);
            
            const inspectBtn = document.createElement('button');
            inspectBtn.className = 'btn secondary';
            inspectBtn.style.marginTop = '8px';
            inspectBtn.style.padding = '4px 8px';
            inspectBtn.style.fontSize = '0.75rem';
            inspectBtn.innerText = 'Test Odor Profile →';
            inspectBtn.onclick = () => inspectCandidate(cand.smiles);
            card.appendChild(inspectBtn);
            
            grid.appendChild(card);
        });
        
    } catch (error) {
        grid.innerHTML = `<p style="color: #ef4444;">Error: ${error.message}</p>`;
    }
}

function inspectCandidate(smiles) {
    goTo('predict-page');
    const input = document.getElementById('mol-input');
    if (input) {
        input.value = smiles;
        predictOdor();
    }
}
