// Progress Tracking Dashboard JavaScript
class ProgressTracker {
    constructor() {
        this.studentId = null;
        this.charts = {};
        this.goals = [];
        this.init();
    }

    init() {
        this.getStudentIdFromURL();
        this.setupEventListeners();
        this.loadProgressData();
    }

    getStudentIdFromURL() {
        const urlParams = new URLSearchParams(window.location.search);
        let studentId = urlParams.get('studentId');

        // If no student ID provided, use a default valid one
        if (!studentId) {
            studentId = 'DV71153'; // Use the requested student ID as default
        }

        this.studentId = studentId;
        document.getElementById('studentIdDisplay').textContent = `Student ID: ${this.studentId}`;
    }

    setupEventListeners() {
        // Refresh button
        document.getElementById('refreshProgress').addEventListener('click', () => {
            this.loadProgressData();
        });

        // Chart filter controls
        document.getElementById('gpaTimeFilter').addEventListener('change', (e) => {
            this.updateGPATrendChart(e.target.value);
        });

        document.getElementById('coursePerformanceFilter').addEventListener('change', (e) => {
            this.updateCoursePerformanceChart(e.target.value);
        });

        document.getElementById('courseTimeFilter').addEventListener('change', (e) => {
            this.loadRecentCourses(e.target.value);
        });

        // Goal management
        document.getElementById('addGoalBtn').addEventListener('click', () => {
            this.showAddGoalModal();
        });

        document.getElementById('closeGoalModal').addEventListener('click', () => {
            this.hideAddGoalModal();
        });

        document.getElementById('cancelGoalBtn').addEventListener('click', () => {
            this.hideAddGoalModal();
        });

        document.getElementById('addGoalForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.addNewGoal();
        });

        // Close modal when clicking outside
        document.getElementById('addGoalModal').addEventListener('click', (e) => {
            if (e.target.id === 'addGoalModal') {
                this.hideAddGoalModal();
            }
        });
    }

    async loadProgressData() {
        this.showLoading();
        
        try {
            // Load all progress data
            await Promise.all([
                this.loadOverviewData(),
                this.loadRecentCourses('current'),
                this.loadAchievements(),
                this.loadGoals(),
                this.loadChartData()
            ]);

            // Initialize charts after data is loaded
            this.initializeCharts();
            
        } catch (error) {
            console.error('Error loading progress data:', error);
            this.showError('Failed to load progress data');
        } finally {
            this.hideLoading();
        }
    }

    async loadOverviewData() {
        try {
            const response = await fetch(`/api/student/${this.studentId}/progress`);
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();

            // Update overview cards with proper data formatting
            document.getElementById('overallGPA').textContent = data.gpa ? data.gpa.toFixed(2) : '--';
            document.getElementById('creditsCompleted').textContent = data.creditsCompleted || '0';
            document.getElementById('coursesPassed').textContent = data.coursesPassed || '0';
            document.getElementById('semesterProgress').textContent = `${data.semesterProgress || 0}%`;

            // Update change indicators
            this.updateChangeIndicator('gpaChange', data.gpaChange);
            this.updateChangeIndicator('creditsChange', data.creditsChange);
            this.updateChangeIndicator('coursesChange', data.coursesChange);
            this.updateChangeIndicator('semesterChange', data.semesterChange);

        } catch (error) {
            console.error('Error loading overview data:', error);
            // Show actual error, don't mask with dummy data
            document.getElementById('overallGPA').textContent = 'Error';
            document.getElementById('creditsCompleted').textContent = 'Error';
            document.getElementById('coursesPassed').textContent = 'Error';
            document.getElementById('semesterProgress').textContent = 'Error';

            // Show the actual error message
            this.showError(`Database error loading progress data: ${error.message}`);
        }
    }

    async loadRecentCourses(timeFilter) {
        try {
            const response = await fetch(`/api/student/${this.studentId}/courses?filter=${timeFilter}`);
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            let courses = await response.json();

            const coursesList = document.getElementById('coursesList');

            // Ensure courses is always an array
            if (!Array.isArray(courses)) {
                console.warn('Courses response is not an array, converting to array:', courses);
                courses = courses && typeof courses === 'object' ? [courses] : [];
            }
            
            if (courses.length === 0) {
                coursesList.innerHTML = `
                    <div class="no-courses">
                        <i class="fas fa-book-open"></i>
                        <p>No courses found for the selected period</p>
                    </div>
                `;
                return;
            }

            coursesList.innerHTML = courses.map(course => `
                <div class="course-item">
                    <div class="course-info">
                        <h4>${course.name || 'Unknown Course'}</h4>
                        <p>${course.code || 'UNKNOWN'} • ${course.credits || 0} credits • ${course.semester || 'Unknown'}</p>
                    </div>
                    <div class="course-grade ${course.gradeClass || 'N/A'}">${course.grade || 'N/A'}</div>
                </div>
            `).join('');

        } catch (error) {
            console.error('Error loading recent courses:', error);
            const coursesList = document.getElementById('coursesList');
            coursesList.innerHTML = `
                <div class="no-courses">
                    <i class="fas fa-exclamation-triangle"></i>
                    <p>Database error loading courses: ${error.message}</p>
                    <p>Cannot load course data from database.</p>
                </div>
            `;
        }
    }

    async loadAchievements() {
        // This would typically fetch from the backend
        // For now, we'll use the static achievements in the HTML
        console.log('Achievements loaded');
    }

    async loadGoals() {
        try {
            const response = await fetch(`/api/student/${this.studentId}/goals`);
            const goals = await response.json();
            this.goals = goals;
            this.renderGoals();
        } catch (error) {
            console.error('Error loading goals:', error);
            // Don't use fake goals - show that goals couldn't be loaded
            this.goals = [];
            const goalsList = document.getElementById('goalsList');
            goalsList.innerHTML = `
                <div class="no-goals">
                    <i class="fas fa-exclamation-triangle"></i>
                    <p>Cannot load goals from database: ${error.message}</p>
                </div>
            `;
        }
    }

    async loadChartData() {
        try {
            const response = await fetch(`/api/student/${this.studentId}/charts`);

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            this.chartData = await response.json();
            console.log('Chart data loaded:', this.chartData);

        } catch (error) {
            console.error('Error loading chart data:', error);
            // Don't use fake chart data - set to null to indicate failure
            this.chartData = null;
            this.showError(`Cannot load chart data from database: ${error.message}`);
        }
    }

    renderGoals() {
        const goalsList = document.getElementById('goalsList');
        goalsList.innerHTML = this.goals.map(goal => {
            const progress = Math.min((goal.current / goal.target) * 100, 100);
            return `
                <div class="goal-item">
                    <div class="goal-content">
                        <h4>${goal.title}</h4>
                        <p>${goal.description}</p>
                        <div class="goal-progress">
                            <div class="progress-bar">
                                <div class="progress-fill" style="width: ${progress}%"></div>
                            </div>
                            <span class="progress-text">${Math.round(progress)}%</span>
                        </div>
                    </div>
                    <div class="goal-actions">
                        <button class="edit-goal-btn" onclick="progressTracker.editGoal(${goal.id})">
                            <i class="fas fa-edit"></i>
                        </button>
                        <button class="delete-goal-btn" onclick="progressTracker.deleteGoal(${goal.id})">
                            <i class="fas fa-trash"></i>
                        </button>
                    </div>
                </div>
            `;
        }).join('');
    }

    initializeCharts() {
        this.initializeGPATrendChart();
        this.initializeCoursePerformanceChart();
        this.initializeCreditProgressChart();
        this.initializeGradeDistributionChart();
    }

    initializeGPATrendChart() {
        const ctx = document.getElementById('gpaTrendChart').getContext('2d');

        // If no chart data available, show error state
        if (!this.chartData?.gpaTrend) {
            ctx.canvas.style.display = 'none';
            const container = ctx.canvas.parentElement;
            container.innerHTML = '<div class="chart-error">No GPA trend data available</div>';
            return;
        }

        const chartData = this.chartData.gpaTrend;

        this.charts.gpaTrend = new Chart(ctx, {
            type: 'line',
            data: {
                labels: chartData.labels,
                datasets: [{
                    label: 'GPA',
                    data: chartData.data,
                    borderColor: '#667eea',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: false,
                        min: 2.0,
                        max: 4.0,
                        ticks: {
                            callback: function(value) {
                                return value.toFixed(1);
                            }
                        }
                    }
                }
            }
        });
    }

    initializeCoursePerformanceChart() {
        const ctx = document.getElementById('coursePerformanceChart').getContext('2d');
        
        this.charts.coursePerformance = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['A', 'B+', 'B', 'B-', 'C+', 'C'],
                datasets: [{
                    label: 'Number of Courses',
                    data: [8, 4, 2, 1, 0, 0],
                    backgroundColor: [
                        '#10b981',
                        '#3b82f6',
                        '#8b5cf6',
                        '#f59e0b',
                        '#ef4444',
                        '#dc2626'
                    ],
                    borderRadius: 8
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            stepSize: 1
                        }
                    }
                }
            }
        });
    }

    initializeCreditProgressChart() {
        const ctx = document.getElementById('creditProgressChart').getContext('2d');

        // If no chart data available, show error state
        if (!this.chartData?.creditProgress) {
            ctx.canvas.style.display = 'none';
            const container = ctx.canvas.parentElement;
            container.innerHTML = '<div class="chart-error">No credit progress data available</div>';
            return;
        }

        const creditData = this.chartData.creditProgress;

        // Only show chart if we have meaningful data
        const chartData = creditData.remaining > 0
            ? [creditData.completed, creditData.remaining]
            : [creditData.completed, 1]; // Show just completed if no remaining data

        const chartLabels = creditData.remaining > 0
            ? ['Completed', 'Remaining']
            : ['Credits Completed'];

        this.charts.creditProgress = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: chartLabels,
                datasets: [{
                    data: chartData,
                    backgroundColor: ['#667eea', '#e5e7eb'],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
    }

    initializeGradeDistributionChart() {
        const ctx = document.getElementById('gradeDistributionChart').getContext('2d');

        // If no chart data available, show error state
        if (!this.chartData?.gradeDistribution) {
            ctx.canvas.style.display = 'none';
            const container = ctx.canvas.parentElement;
            container.innerHTML = '<div class="chart-error">No grade distribution data available</div>';
            return;
        }

        const gradeData = this.chartData.gradeDistribution;

        this.charts.gradeDistribution = new Chart(ctx, {
            type: 'pie',
            data: {
                labels: gradeData.labels.map(label => `${label} Range`),
                datasets: [{
                    data: gradeData.data,
                    backgroundColor: [
                        '#10b981',
                        '#3b82f6',
                        '#f59e0b',
                        '#ef4444'
                    ]
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
    }

    updateGPATrendChart(filter) {
        // This would update the chart based on the selected filter
        console.log('Updating GPA trend chart with filter:', filter);
    }

    updateCoursePerformanceChart(filter) {
        // This would update the chart based on the selected filter
        console.log('Updating course performance chart with filter:', filter);
    }

    updateChangeIndicator(elementId, change) {
        const element = document.getElementById(elementId);
        if (!element) return;

        if (change > 0) {
            element.textContent = `+${change}%`;
            element.className = 'metric-change positive';
        } else if (change < 0) {
            element.textContent = `${change}%`;
            element.className = 'metric-change negative';
        } else {
            element.textContent = '0%';
            element.className = 'metric-change';
        }
    }

    showAddGoalModal() {
        document.getElementById('addGoalModal').style.display = 'block';
    }

    hideAddGoalModal() {
        document.getElementById('addGoalModal').style.display = 'none';
        document.getElementById('addGoalForm').reset();
    }

    addNewGoal() {
        const formData = new FormData(document.getElementById('addGoalForm'));
        const goal = {
            id: Date.now(),
            title: formData.get('goalTitle'),
            description: formData.get('goalDescription'),
            target: parseFloat(formData.get('goalTarget')),
            current: 0,
            type: formData.get('goalType')
        };

        this.goals.push(goal);
        this.renderGoals();
        this.hideAddGoalModal();

        // In a real application, you would save this to the backend
        console.log('New goal added:', goal);
    }

    editGoal(goalId) {
        const goal = this.goals.find(g => g.id === goalId);
        if (goal) {
            // In a real application, you would show an edit modal
            console.log('Editing goal:', goal);
        }
    }

    deleteGoal(goalId) {
        if (confirm('Are you sure you want to delete this goal?')) {
            this.goals = this.goals.filter(g => g.id !== goalId);
            this.renderGoals();
            console.log('Goal deleted:', goalId);
        }
    }

    showLoading() {
        document.getElementById('loadingOverlay').style.display = 'block';
    }

    hideLoading() {
        document.getElementById('loadingOverlay').style.display = 'none';
    }

    showError(message) {
        // Simple error display - in a real app you might want a more sophisticated error system
        alert(`Error: ${message}`);
    }
}

// Utility function to go back
function goBack() {
    window.history.back();
}

// Initialize the progress tracker when the page loads
let progressTracker;
document.addEventListener('DOMContentLoaded', function() {
    progressTracker = new ProgressTracker();
});
