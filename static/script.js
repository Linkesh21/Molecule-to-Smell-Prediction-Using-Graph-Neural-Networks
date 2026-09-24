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
        const smiles = jsmeApplet.smiles();
        document.getElementById('mol-input').value = smiles;
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
    list.innerHTML = '<li>Analyzing molecule...</li>';

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
            
            // Image
            const img = document.createElement('img');
            img.src = `https://cactus.nci.nih.gov/chemical/structure/${encodeURIComponent(cand.smiles)}/image`;
            img.onerror = () => { img.style.display = 'none'; }; // Hide if image fails to load
            
            card.appendChild(img);
            
            const p = document.createElement('p');
            p.innerText = cand.smiles;
            card.appendChild(p);
            
            const score = document.createElement('small');
            score.innerText = `Match Score: ${cand.score.toFixed(2)}`;
            card.appendChild(score);
            
            grid.appendChild(card);
        });
        
    } catch (error) {
        grid.innerHTML = `<p style="color: #ef4444;">Error: ${error.message}</p>`;
    }
}
