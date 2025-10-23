// JavaScript for NVD Severity Classifier Dashboard

class SeverityDashboard {
    constructor() {
        this.allResults = [];
        this.filteredResults = [];
        this.displayedResults = [];
        this.currentPage = 1;
        this.resultsPerPage = 50;
        this.currentTaskId = null;
        this.progressInterval = null;
        this.recentActivity = [];
        
        this.initEventListeners();
        this.updateStats();
        this.updateResultsInfo();
    }

    initEventListeners() {
        // Single text prediction form
        document.getElementById('singlePredictionForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.analyzeSingleText();
        });

        // Batch prediction form
        document.getElementById('batchPredictionForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.startBatchProcessing();
        });

        // Stop processing button
        document.getElementById('stopButton').addEventListener('click', () => {
            this.stopProcessing();
        });

        // Search and filter
        document.getElementById('searchInput').addEventListener('input', () => {
            this.filterAndSortResults();
        });

        document.getElementById('severityFilter').addEventListener('change', () => {
            this.filterAndSortResults();
        });

        document.getElementById('sortBy').addEventListener('change', () => {
            this.filterAndSortResults();
        });

        // Clear results
        document.getElementById('clearResults').addEventListener('click', () => {
            this.clearAllResults();
        });
    }

    async analyzeSingleText() {
        const text = document.getElementById('vulnerabilityText').value.trim();
        
        if (!text) {
            this.showAlert('Please enter a vulnerability description', 'warning');
            return;
        }

        this.showLoading(true);

        try {
            const response = await fetch('/predict_single', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: text })
            });

            const result = await response.json();

            if (result.success) {
                this.addSingleResult(result);
                this.addRecentActivity(`Analyzed: "${text.substring(0, 50)}..."`);
                document.getElementById('vulnerabilityText').value = '';
                this.showAlert('Analysis completed successfully!', 'success');
            } else {
                this.showAlert(result.error || 'Prediction failed', 'danger');
            }
        } catch (error) {
            this.showAlert('Network error: ' + error.message, 'danger');
        } finally {
            this.showLoading(false);
        }
    }

    async startBatchProcessing() {
        const fileInput = document.getElementById('jsonFile');
        const file = fileInput.files[0];

        if (!file) {
            this.showAlert('Please select a JSON file', 'warning');
            return;
        }

        this.showProgressSection(true);
        this.showLoading(true);

        try {
            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch('/start_batch_processing', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (result.success) {
                this.currentTaskId = result.task_id;
                this.startProgressTracking();
                this.showAlert(`Started processing ${result.total_vulnerabilities} vulnerabilities`, 'info');
                this.addRecentActivity(`Started processing: ${file.name} (${result.total_vulnerabilities} vulnerabilities)`);
            } else {
                this.showAlert(result.error || 'Failed to start processing', 'danger');
                this.showProgressSection(false);
            }
        } catch (error) {
            this.showAlert('Network error: ' + error.message, 'danger');
            this.showProgressSection(false);
        } finally {
            this.showLoading(false);
        }
    }

    startProgressTracking() {
        this.progressInterval = setInterval(async () => {
            if (!this.currentTaskId) return;

            try {
                const response = await fetch(`/get_processing_status/${this.currentTaskId}`);
                const status = await response.json();

                if (status.status === 'processing' || status.status === 'completed') {
                    this.updateProgress(status);
                    
                    if (status.results && status.results.length > 0) {
                        // Add new results to allResults
                        const newResults = status.results.filter(newResult => 
                            !this.allResults.some(existingResult => 
                                existingResult.cve_id === newResult.cve_id && 
                                existingResult.text === newResult.text
                            )
                        );
                        
                        this.allResults.push(...newResults);
                        this.filterAndSortResults();
                        this.updateStats();
                    }

                    if (status.status === 'completed') {
                        this.processingCompleted(status);
                    }
                } else if (status.status === 'stopped') {
                    this.showAlert('Processing stopped by user', 'warning');
                    this.processingStopped(status);
                } else if (status.status === 'error') {
                    this.showAlert('Processing error: ' + status.error, 'danger');
                    this.processingStopped(status);
                }
            } catch (error) {
                console.error('Progress tracking error:', error);
            }
        }, 1000);
    }

    updateProgress(status) {
        const progressBar = document.getElementById('progressBar');
        const progressText = document.getElementById('progressText');
        
        const percentage = (status.processed / status.total) * 100;
        progressBar.style.width = `${percentage}%`;
        progressText.textContent = `Processing: ${status.processed}/${status.total} (${percentage.toFixed(1)}%)`;
    }

    processingCompleted(status) {
        this.clearProgressTracking();
        this.showProgressSection(false);
        this.showAlert(`Processing completed! ${status.processed} vulnerabilities analyzed`, 'success');
        this.addRecentActivity(`Completed: Processed ${status.processed} vulnerabilities`);
    }

    processingStopped(status) {
        this.clearProgressTracking();
        this.showProgressSection(false);
        this.showAlert(`Processing stopped. Processed ${status.processed} vulnerabilities`, 'info');
        this.addRecentActivity(`Stopped: Processed ${status.processed} vulnerabilities`);
    }

    async stopProcessing() {
        if (!this.currentTaskId) return;

        try {
            await fetch(`/stop_processing/${this.currentTaskId}`, { method: 'POST' });
        } catch (error) {
            console.error('Stop processing error:', error);
        }
    }

    clearProgressTracking() {
        if (this.progressInterval) {
            clearInterval(this.progressInterval);
            this.progressInterval = null;
        }
        this.currentTaskId = null;
    }

    showProgressSection(show) {
        const progressSection = document.getElementById('progressSection');
        const processButton = document.getElementById('processButton');
        
        if (show) {
            progressSection.style.display = 'block';
            processButton.disabled = true;
        } else {
            progressSection.style.display = 'none';
            processButton.disabled = false;
            
            // Reset progress bar
            document.getElementById('progressBar').style.width = '0%';
            document.getElementById('progressText').textContent = 'Processing: 0/0';
        }
    }

    addSingleResult(result) {
        const newResult = {
            ...result,
            cve_id: 'Manual Input',
            product: result.product || 'Manual Input',
            priority: this.getSeverityPriority(result.severity)
        };
        
        this.allResults.unshift(newResult);
        this.filterAndSortResults();
        this.updateStats();
    }

    filterAndSortResults() {
        const searchTerm = document.getElementById('searchInput').value.toLowerCase();
        const severityFilter = document.getElementById('severityFilter').value;
        const sortBy = document.getElementById('sortBy').value;

        // Filter results
        this.filteredResults = this.allResults.filter(result => {
            const matchesSearch = !searchTerm || 
                (result.cve_id && result.cve_id.toLowerCase().includes(searchTerm)) ||
                (result.product && result.product.toLowerCase().includes(searchTerm)) ||
                (result.text && result.text.toLowerCase().includes(searchTerm));
            
            const matchesSeverity = !severityFilter || result.severity === severityFilter;
            
            return matchesSearch && matchesSeverity;
        });

        // Sort results
        this.filteredResults.sort((a, b) => {
            switch (sortBy) {
                case 'severity':
                    const severityOrder = {'High': 0, 'Medium': 1, 'Low': 2};
                    return severityOrder[a.severity] - severityOrder[b.severity];
                case 'confidence':
                    return b.confidence - a.confidence;
                case 'product':
                    return (a.product || '').localeCompare(b.product || '');
                case 'cve_id':
                    return (a.cve_id || '').localeCompare(b.cve_id || '');
                case 'priority':
                default:
                    return (a.priority || 999) - (b.priority || 999);
            }
        });

        this.currentPage = 1;
        this.updatePagination();
        this.updateDisplayedResults();
    }

    updateDisplayedResults() {
        const startIndex = (this.currentPage - 1) * this.resultsPerPage;
        const endIndex = startIndex + this.resultsPerPage;
        this.displayedResults = this.filteredResults.slice(startIndex, endIndex);
        
        this.updateResultsTable();
        this.updateResultsInfo();
    }

    updatePagination() {
        const totalPages = Math.ceil(this.filteredResults.length / this.resultsPerPage);
        const pagination = document.getElementById('pagination');
        const paginationNav = document.getElementById('paginationNav');

        if (totalPages <= 1) {
            paginationNav.style.display = 'none';
            return;
        }

        paginationNav.style.display = 'block';
        
        let paginationHTML = '';
        
        // Previous button
        paginationHTML += `
            <li class="page-item ${this.currentPage === 1 ? 'disabled' : ''}">
                <a class="page-link" href="#" data-page="${this.currentPage - 1}">Previous</a>
            </li>
        `;
        
        // Page numbers
        for (let i = 1; i <= totalPages; i++) {
            if (i === 1 || i === totalPages || (i >= this.currentPage - 2 && i <= this.currentPage + 2)) {
                paginationHTML += `
                    <li class="page-item ${i === this.currentPage ? 'active' : ''}">
                        <a class="page-link" href="#" data-page="${i}">${i}</a>
                    </li>
                `;
            } else if (i === this.currentPage - 3 || i === this.currentPage + 3) {
                paginationHTML += `<li class="page-item disabled"><span class="page-link">...</span></li>`;
            }
        }
        
        // Next button
        paginationHTML += `
            <li class="page-item ${this.currentPage === totalPages ? 'disabled' : ''}">
                <a class="page-link" href="#" data-page="${this.currentPage + 1}">Next</a>
            </li>
        `;
        
        pagination.innerHTML = paginationHTML;
        
        // Add event listeners to pagination links
        pagination.querySelectorAll('.page-link').forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const page = parseInt(link.getAttribute('data-page'));
                if (page && page !== this.currentPage) {
                    this.currentPage = page;
                    this.updateDisplayedResults();
                    this.updatePagination();
                }
            });
        });
    }

    updateResultsTable() {
        const tbody = document.getElementById('resultsBody');
        
        if (this.displayedResults.length === 0) {
            tbody.innerHTML = `
                <tr>
                    <td colspan="7" class="text-center text-muted">
                        ${this.allResults.length === 0 ? 
                            'No results yet. Analyze a vulnerability to see results.' : 
                            'No results match your search criteria.'}
                    </td>
                </tr>
            `;
            return;
        }

        tbody.innerHTML = this.displayedResults.map((result, index) => {
            const globalIndex = (this.currentPage - 1) * this.resultsPerPage + index + 1;
            return `
                <tr>
                    <td>
                        <div class="priority-badge priority-${this.getSeverityPriority(result.severity)}">
                            ${globalIndex}
                        </div>
                    </td>
                    <td>
                        <strong>${result.cve_id || 'N/A'}</strong>
                    </td>
                    <td>
                        <span class="badge bg-secondary product-badge">${result.product || 'Unknown'}</span>
                    </td>
                    <td>
                        <div class="vulnerability-desc" title="${result.text}">
                            ${this.truncateText(result.text, 100)}
                        </div>
                    </td>
                    <td>
                        <span class="severity-badge severity-${result.severity.toLowerCase()}">
                            ${result.severity}
                        </span>
                    </td>
                    <td>
                        <div class="d-flex align-items-center">
                            <span class="me-2">${result.confidence}%</span>
                            <div class="confidence-bar" style="width: 60px;">
                                <div class="confidence-fill confidence-${this.getConfidenceLevel(result.confidence)}" 
                                     style="width: ${result.confidence}%"></div>
                            </div>
                        </div>
                    </td>
                    <td>
                        <small class="text-muted">${result.timestamp}</small>
                    </td>
                </tr>
            `;
        }).join('');
    }

    updateStats() {
        const counts = {
            High: this.allResults.filter(r => r.severity === 'High').length,
            Medium: this.allResults.filter(r => r.severity === 'Medium').length,
            Low: this.allResults.filter(r => r.severity === 'Low').length
        };
        
        document.getElementById('highCount').textContent = counts.High;
        document.getElementById('mediumCount').textContent = counts.Medium;
        document.getElementById('lowCount').textContent = counts.Low;
    }

    updateResultsInfo() {
        const showingResults = document.getElementById('showingResults');
        const resultsCount = document.getElementById('resultsCount');
        
        const start = this.displayedResults.length > 0 ? (this.currentPage - 1) * this.resultsPerPage + 1 : 0;
        const end = start + this.displayedResults.length - 1;
        const total = this.filteredResults.length;
        const allTotal = this.allResults.length;
        
        if (this.displayedResults.length > 0) {
            showingResults.textContent = `Showing ${start}-${end} of ${total}`;
        } else {
            showingResults.textContent = `Showing 0 of ${total}`;
        }
        resultsCount.textContent = `${allTotal} total results`;
    }

    addRecentActivity(message) {
        this.recentActivity.unshift({
            message: message,
            timestamp: new Date().toLocaleTimeString()
        });

        // Keep only last 5 activities
        this.recentActivity = this.recentActivity.slice(0, 5);
        this.updateRecentActivity();
    }

    updateRecentActivity() {
        const container = document.getElementById('recentActivity');
        
        if (this.recentActivity.length === 0) {
            container.innerHTML = '<p class="text-muted">No recent activity</p>';
            return;
        }

        container.innerHTML = this.recentActivity.map(activity => `
            <div class="recent-activity-item">
                <div class="d-flex justify-content-between">
                    <span class="activity-message">${activity.message}</span>
                    <small class="text-muted">${activity.timestamp}</small>
                </div>
            </div>
        `).join('');
    }

    getSeverityPriority(severity) {
        const priorityMap = {
            'High': 1,
            'Medium': 2,
            'Low': 3
        };
        return priorityMap[severity] || 3;
    }

    getConfidenceLevel(confidence) {
        if (confidence >= 80) return 'high';
        if (confidence >= 60) return 'medium';
        return 'low';
    }

    truncateText(text, maxLength) {
        if (!text) return 'No description';
        if (text.length <= maxLength) return text;
        return text.substring(0, maxLength) + '...';
    }

    showLoading(show) {
        const spinner = document.getElementById('loadingSpinner');
        spinner.style.display = show ? 'block' : 'none';
    }

    showAlert(message, type = 'info') {
        const alertContainer = document.getElementById('alertContainer');
        const alertId = 'alert-' + Date.now();
        
        const alert = document.createElement('div');
        alert.id = alertId;
        alert.className = `alert alert-${type} alert-dismissible fade show`;
        alert.innerHTML = `
            <i class="fas fa-${this.getAlertIcon(type)} me-2"></i>
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;

        if (!alertContainer) {
            // Create alert container if it doesn't exist
            const container = document.createElement('div');
            container.id = 'alertContainer';
            container.className = 'position-fixed top-0 end-0 p-3';
            container.style.zIndex = '1050';
            document.body.appendChild(container);
        }

        document.getElementById('alertContainer').appendChild(alert);

        // Auto remove after 5 seconds
        setTimeout(() => {
            const alertElement = document.getElementById(alertId);
            if (alertElement) {
                alertElement.remove();
            }
        }, 5000);
    }

    getAlertIcon(type) {
        const icons = {
            'success': 'check-circle',
            'danger': 'exclamation-triangle',
            'warning': 'exclamation-circle',
            'info': 'info-circle'
        };
        return icons[type] || 'info-circle';
    }

    clearAllResults() {
        this.allResults = [];
        this.filteredResults = [];
        this.displayedResults = [];
        this.currentPage = 1;
        
        this.updateResultsTable();
        this.updateStats();
        this.updatePagination();
        this.updateResultsInfo();
        
        // Clear on server side too
        fetch('/clear_results', { method: 'POST' }).catch(console.error);
        
        this.showAlert('All results cleared', 'info');
        this.addRecentActivity('Cleared all results');
    }
}

// Initialize dashboard when page loads
document.addEventListener('DOMContentLoaded', () => {
    new SeverityDashboard();
});