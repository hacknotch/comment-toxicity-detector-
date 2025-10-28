const btn = document.getElementById("analyze");
const visualBtn = document.getElementById("visual-analysis");
const securityBtn = document.getElementById("security-analysis");
const textarea = document.getElementById("comment");
const result = document.getElementById("result");
const labelEl = document.getElementById("label");
const confEl = document.getElementById("confidence");
const visualResults = document.getElementById("visual-results");
const analysisContent = document.getElementById("analysis-content");
const securityResults = document.getElementById("security-results");
const securityContent = document.getElementById("security-content");

async function predict(text) {
	const res = await fetch("http://127.0.0.1:5000/predict", {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify({ comment: text })
	});
	if (!res.ok) {
		const err = await res.json().catch(() => ({}));
		throw new Error(err.error || `HTTP ${res.status}`);
	}
	return res.json();
}

async function analyzeVisual(text) {
	const res = await fetch("http://127.0.0.1:5000/analyze_visual", {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify({ comment: text })
	});
	if (!res.ok) {
		const err = await res.json().catch(() => ({}));
		throw new Error(err.error || `HTTP ${res.status}`);
	}
	return res.json();
}

async function analyzeSecurity(text) {
	const res = await fetch("http://127.0.0.1:5000/analyze_security", {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify({ comment: text })
	});
	if (!res.ok) {
		const err = await res.json().catch(() => ({}));
		throw new Error(err.error || `HTTP ${res.status}`);
	}
	return res.json();
}

function colorBadge(conf) {
	// 🟢 non-toxic (<0.4), 🟠 mild (0.4–0.7), 🔴 toxic (>0.7)
	labelEl.classList.remove("green", "orange", "red");
	if (conf < 0.4) {
		labelEl.classList.add("green");
	} else if (conf <= 0.7) {
		labelEl.classList.add("orange");
	} else {
		labelEl.classList.add("red");
	}
}

function renderVisualAnalysis(data) {
	const scores = data.scores || {};
	const overallToxicity = data.overall_toxicity || 0;
	const severity = data.severity || "low";
	const toxicWords = data.toxic_words || [];
	const patternMatches = data.pattern_matches || 0;
	const exclamationCount = data.exclamation_count || 0;
	const capsRatio = data.caps_ratio || 0;
	const repeatedChars = data.repeated_chars || 0;
	
	analysisContent.innerHTML = `
		<div class="analysis-grid">
			<div class="analysis-item">
				<h4>Word Count</h4>
				<div class="value">${data.word_count || 0}</div>
			</div>
			<div class="analysis-item">
				<h4>Character Count</h4>
				<div class="value">${data.char_count || 0}</div>
			</div>
			<div class="analysis-item">
				<h4>Overall Toxicity</h4>
				<div class="value">${(overallToxicity * 100).toFixed(1)}%</div>
				<div class="score-bar">
					<div class="score-fill" style="width: ${overallToxicity * 100}%"></div>
				</div>
			</div>
			<div class="analysis-item">
				<h4>Severity Level</h4>
				<div class="value" style="color: ${severity === 'high' ? 'var(--red)' : severity === 'medium' ? 'var(--orange)' : 'var(--green)'}">
					${severity.toUpperCase()}
				</div>
			</div>
		</div>
		<div class="detailed-scores">
			<h4>Detailed Scores</h4>
			${Object.entries(scores).map(([label, score]) => `
				<div class="score-item">
					<span class="score-label">${label.charAt(0).toUpperCase() + label.slice(1)}</span>
					<div class="score-bar">
						<div class="score-fill" style="width: ${score * 100}%"></div>
					</div>
					<span class="score-value">${(score * 100).toFixed(1)}%</span>
				</div>
			`).join('')}
		</div>
		${toxicWords.length > 0 ? `
			<div class="toxic-words">
				<h4>Detected Toxic Words</h4>
				<div class="word-list">${toxicWords.map(word => `<span class="toxic-word">${word}</span>`).join(', ')}</div>
			</div>
		` : ''}
		${patternMatches > 0 ? `
			<div class="pattern-matches">
				<h4>Harsh Patterns Detected</h4>
				<div class="value">${patternMatches} pattern(s) found</div>
			</div>
		` : ''}
		${exclamationCount > 0 ? `
			<div class="aggressive-indicators">
				<h4>Aggressive Indicators</h4>
				<div class="indicator-item">
					<span>Exclamation marks: ${exclamationCount}</span>
				</div>
				${capsRatio > 0.5 ? `
					<div class="indicator-item">
						<span>Excessive caps: ${(capsRatio * 100).toFixed(1)}%</span>
					</div>
				` : ''}
				${repeatedChars > 0 ? `
					<div class="indicator-item">
						<span>Repeated characters: ${repeatedChars}</span>
					</div>
				` : ''}
			</div>
		` : ''}
	`;
}

function renderSecurityAnalysis(data) {
	const threatCount = data.threat_count || 0;
	const fraudCount = data.fraud_count || 0;
	const totalThreats = data.total_threats || 0;
	const riskLevel = data.risk_level || "LOW";
	const threatCategories = data.threat_categories || {};
	const fraudIndicators = data.fraud_indicators || [];
	const fraudScore = data.fraud_score || 0;
	
	securityContent.innerHTML = `
		<div class="analysis-grid">
			<div class="analysis-item">
				<h4>Word Count</h4>
				<div class="value">${data.word_count || 0}</div>
			</div>
			<div class="analysis-item">
				<h4>Character Count</h4>
				<div class="value">${data.char_count || 0}</div>
			</div>
			<div class="analysis-item">
				<h4>Security Threats</h4>
				<div class="value">${threatCount}</div>
			</div>
			<div class="analysis-item">
				<h4>Fraud Indicators</h4>
				<div class="value">${fraudCount}</div>
			</div>
			<div class="analysis-item">
				<h4>Total Threats</h4>
				<div class="value">${totalThreats}</div>
			</div>
			<div class="analysis-item">
				<h4>Risk Level</h4>
				<div class="value" style="color: ${riskLevel === 'CRITICAL' ? 'var(--red)' : riskLevel === 'HIGH' ? 'var(--red)' : riskLevel === 'MEDIUM' ? 'var(--orange)' : 'var(--green)'}">
					${riskLevel}
				</div>
			</div>
		</div>
		${totalThreats > 0 ? `
			<div class="threat-categories">
				<h4>Security Analysis Results</h4>
				${threatCount > 0 ? `
					<div class="threat-category">
						<h5>🔒 SECURITY THREATS (${threatCount})</h5>
						<div class="threat-list">${data.found_threats ? data.found_threats.length + ' threat(s) detected' : 'No specific threats'}</div>
					</div>
				` : ''}
				${fraudCount > 0 ? `
					<div class="threat-category">
						<h5>💰 FRAUD INDICATORS (${fraudCount})</h5>
						<div class="threat-list">Fraud Score: ${(fraudScore * 100).toFixed(1)}%</div>
						<div class="threat-list">${fraudIndicators.length} fraud pattern(s) detected</div>
					</div>
				` : ''}
				${Object.entries(threatCategories).map(([category, threats]) => 
					threats.length > 0 ? `
						<div class="threat-category">
							<h5>${category.replace('_', ' ').toUpperCase()} (${threats.length})</h5>
							<div class="threat-list">${threats.join(', ')}</div>
						</div>
					` : ''
				).join('')}
			</div>
		` : `
			<div class="no-threats">
				<h4>✅ No Security Threats Detected</h4>
				<p>This content appears to be safe from security threats and fraud.</p>
			</div>
		`}
	`;
}

btn.addEventListener("click", async () => {
	const text = textarea.value.trim();
	if (!text) return;
	btn.disabled = true;
	btn.textContent = "Analyzing...";
	try {
		const data = await predict(text);
		labelEl.textContent = data.label;
		colorBadge(data.confidence);
		confEl.textContent = `confidence: ${data.confidence.toFixed(2)}`;
		result.classList.remove("hidden");
	} catch (e) {
		labelEl.textContent = "error";
		labelEl.classList.remove("green", "orange", "red");
		confEl.textContent = e.message;
		result.classList.remove("hidden");
	} finally {
		btn.disabled = false;
		btn.textContent = "Analyze";
	}
});

visualBtn.addEventListener("click", async () => {
	const text = textarea.value.trim();
	if (!text) return;
	visualBtn.disabled = true;
	visualBtn.textContent = "Analyzing...";
	try {
		const data = await analyzeVisual(text);
		renderVisualAnalysis(data);
		visualResults.classList.remove("hidden");
	} catch (e) {
		analysisContent.innerHTML = `<p style="color: var(--red)">Error: ${e.message}</p>`;
		visualResults.classList.remove("hidden");
	} finally {
		visualBtn.disabled = false;
		visualBtn.textContent = "Visual Analysis";
	}
});

securityBtn.addEventListener("click", async () => {
	const text = textarea.value.trim();
	if (!text) return;
	securityBtn.disabled = true;
	securityBtn.textContent = "Analyzing...";
	try {
		const data = await analyzeSecurity(text);
		renderSecurityAnalysis(data);
		securityResults.classList.remove("hidden");
	} catch (e) {
		securityContent.innerHTML = `<p style="color: var(--red)">Error: ${e.message}</p>`;
		securityResults.classList.remove("hidden");
	} finally {
		securityBtn.disabled = false;
		securityBtn.textContent = "Security Analysis";
	}
});
