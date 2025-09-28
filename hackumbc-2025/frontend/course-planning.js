// Course Planning JavaScript - Using Real Neo4j Data
class CoursePlanning {
    constructor() {
        this.studentId = this.getStudentIdFromUrl();
        this.selectedTerm = null;
        this.selectedCourses = [];
        this.studentData = null;
        this.availableTerms = [];
        this.availableCourses = [];

        this.initializeApplication();
    }

    getStudentIdFromUrl() {
        const urlParams = new URLSearchParams(window.location.search);
        return urlParams.get('studentId');
    }

    async initializeApplication() {
        if (!this.studentId) {
            this.showNotification('No student ID provided. Redirecting to home page.', 'error');
            setTimeout(() => {
                window.location.href = '/';
            }, 2000);
            return;
        }

        // Display student ID
        const studentDisplayElement = document.getElementById('studentDisplayId');
        if (studentDisplayElement) {
            studentDisplayElement.textContent = this.studentId;
        }

        // Load student data from Neo4j
        await this.loadStudentData();

        // Load available terms
        await this.loadAvailableTerms();

        // Initialize event listeners
        this.initializeEventListeners();
        
        // Show helpful instruction
        this.showNotification('Please select a term below to see your personalized course recommendations!', 'info');
    }

    async loadStudentData() {
        try {
            this.showLoading('Loading student information from database...');

            // Fetch student data using Neo4j queries
            const response = await fetch(`/api/student-info/${this.studentId}`);
            if (!response.ok) {
                throw new Error('Failed to load student data');
            }

            this.studentData = await response.json();
            this.displayStudentInfo();

        } catch (error) {
            console.error('Error loading student data:', error);
            this.showNotification('Error loading student data. Please try again.', 'error');
        } finally {
            this.hideLoading();
        }
    }

    displayStudentInfo() {
        if (!this.studentData) return;

        // Update current status
        document.getElementById('enrollmentDate').textContent = this.studentData.enrollment_date || 'Not specified';
        document.getElementById('expectedGraduation').textContent = this.studentData.expected_graduation || 'Not specified';
        document.getElementById('currentDegree').textContent = this.studentData.degree || 'Not specified';
        document.getElementById('completedCount').textContent = this.studentData.completed_courses_count || '0';

        // Update preferences
        document.getElementById('learningStyle').textContent = this.studentData.learning_style || 'Not specified';
        document.getElementById('preferredCourseLoad').textContent = `${this.studentData.preferred_course_load || 'Not specified'} credits per semester`;
        document.getElementById('instructionMode').textContent = this.studentData.instruction_mode || 'Not specified';
        document.getElementById('preferredPace').textContent = this.studentData.preferred_pace || 'Not specified';
    }

    async loadAvailableTerms() {
        try {
            const response = await fetch('/api/terms');
            if (!response.ok) {
                throw new Error('Failed to load terms');
            }

            this.availableTerms = await response.json();
            this.populateTermDropdown();
        } catch (error) {
            console.error('Error loading terms:', error);
            this.showNotification('Error loading terms. Please try again.', 'error');
        }
    }

    populateTermDropdown() {
        const termSelect = document.getElementById('termSelect');
        termSelect.innerHTML = '<option value="">Choose a term...</option>';

        this.availableTerms.forEach(term => {
            const option = document.createElement('option');
            option.value = term.id;
            option.textContent = term.name;
            termSelect.appendChild(option);
        });
    }

    initializeEventListeners() {
        // Term dropdown change
        document.getElementById('termSelect')?.addEventListener('change', (e) => {
            const getRecommendationsBtn = document.getElementById('getRecommendationsBtn');
            getRecommendationsBtn.disabled = !e.target.value;
        });

        // Get recommendations button
        document.getElementById('getRecommendationsBtn')?.addEventListener('click', async () => {
            const termSelect = document.getElementById('termSelect');
            const selectedTermId = termSelect.value;
            const selectedTermName = termSelect.options[termSelect.selectedIndex].textContent;
            
            if (selectedTermId) {
                await this.selectTerm(selectedTermId, selectedTermName);
            }
        });

        // Course selection
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('add-course-btn')) {
                const courseId = e.target.getAttribute('data-course-id');
                this.addCourse(courseId);
            }
            if (e.target.classList.contains('remove-course-btn')) {
                const courseId = e.target.getAttribute('data-course-id');
                this.removeCourse(courseId);
            }
        });

        // Filter changes
        document.getElementById('showPrereqOnly')?.addEventListener('change', () => {
            this.filterCourses();
        });
        document.getElementById('respectCourseLoad')?.addEventListener('change', () => {
            this.filterCourses();
        });
        document.getElementById('matchLearningStyle')?.addEventListener('change', () => {
            this.filterCourses();
        });

        // Export plan
        document.getElementById('exportPlanBtn')?.addEventListener('click', () => {
            this.exportCoursePlan();
        });
    }

    async selectTerm(termId, termName) {
        this.selectedTerm = { id: termId, name: termName };

        // Update UI
        document.getElementById('selectedTermDisplay').textContent = termName;
        document.getElementById('courseRecommendations').classList.remove('hidden');

        // Scroll to recommendations
        document.getElementById('courseRecommendations').scrollIntoView({ behavior: 'smooth' });

        // Load course recommendations
        await this.loadCourseRecommendations();
    }

    async loadCourseRecommendations() {
        try {
            this.showLoading('Generating course recommendations from Neo4j...');

            const response = await fetch('/api/course-recommendations', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    student_id: this.studentId,
                    term_id: this.selectedTerm.id
                })
            });

            if (!response.ok) {
                throw new Error('Failed to load recommendations');
            }

            this.availableCourses = await response.json();
            this.displayCourseRecommendations();

        } catch (error) {
            console.error('Error loading recommendations:', error);
            this.showNotification('Error loading course recommendations. Please try again.', 'error');
        } finally {
            this.hideLoading();
        }
    }

    displayCourseRecommendations() {
        const container = document.getElementById('recommendedCourses');
        container.innerHTML = '';

        if (!this.availableCourses || this.availableCourses.length === 0) {
            container.innerHTML = '<p class="no-courses">No courses available for this term.</p>';
            return;
        }

        // Sort by priority score (if available) or alphabetically
        const sortedCourses = [...this.availableCourses].sort((a, b) => {
            if (a.priority_score && b.priority_score) {
                return b.priority_score - a.priority_score;
            }
            return a.course_id.localeCompare(b.course_id);
        });

        sortedCourses.forEach(course => {
            const courseCard = document.createElement('div');
            courseCard.className = `course-card ${course.is_blocked ? 'blocked' : ''}`;

            const priorityScore = course.priority_score || 0;
            const compatibilityScore = course.faculty_compatibility || 0;

            const difficultyColor = this.getDifficultyColor(course.difficulty);
            const compatibilityColor = this.getCompatibilityColor(compatibilityScore);

            courseCard.innerHTML = `
                <div class="course-header">
                    <h4>${course.course_id}</h4>
                    ${priorityScore > 0 ? `<span class="priority-score">${priorityScore}% match</span>` : ''}
                </div>
                <h5>${course.course_name || course.course_id}</h5>
                <div class="course-details">
                    <span class="credits">${course.credits || 3} credits</span>
                    ${course.difficulty ? `<span class="difficulty" style="color: ${difficultyColor}">${course.difficulty}</span>` : ''}
                    <span class="instruction-mode ${course.instruction_mode_compatibility >= 80 ? 'compatible' : course.instruction_mode_compatibility >= 60 ? 'partial' : course.instruction_mode_compatibility !== null ? 'incompatible' : ''}">
                        ${course.instruction_mode || 'Not specified'}
                        ${course.instruction_mode_compatibility !== null ? `<span class="mode-compatibility">${course.instruction_mode_compatibility}% match</span>` : ''}
                    </span>
                    ${course.is_prerequisite ? '<span class="prerequisite-badge">Prerequisite</span>' : ''}
                </div>
                
                ${course.faculty_options && course.faculty_options.length > 0 ? `
                    <div class="faculty-info">
                        <strong>Faculty Options:</strong>
                        <div class="faculty-list">
                            ${course.faculty_options.map(faculty => `
                                <div class="faculty-option">
                                    <span class="faculty-name">${faculty.name || 'TBA'}</span>
                                    <span class="faculty-style">${faculty.teachingStyle || 'Not specified'}</span>
                                    <span class="compatibility" style="color: ${this.getCompatibilityColor(faculty.compatibility)}">
                                        ${faculty.compatibility}% match
                                    </span>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                ` : ''}
                
                ${course.is_prerequisite && course.prerequisite_for_names && course.prerequisite_for_names.length > 0 ? `
                    <div class="prerequisite-info">
                        <strong>Required for:</strong>
                        <div class="prerequisite-list">
                            ${course.prerequisite_for_names.map((name, index) => `
                                <span class="prerequisite-course">${course.prerequisite_for[index]} - ${name}</span>
                            `).join(', ')}
                        </div>
                    </div>
                ` : ''}
                
                <p class="recommendation-reason">${course.recommendation_reason || 'Available for registration'}</p>
                
                ${course.missing_prerequisites && course.missing_prerequisites.length > 0 ? `
                    <div class="missing-prereqs">
                        <strong>Missing Prerequisites:</strong>
                        <div class="missing-prereq-list">
                            ${course.missing_prerequisites.map((prereq, index) => `
                                <span class="missing-prereq">${prereq} - ${course.missing_prerequisite_names[index] || prereq}</span>
                            `).join(', ')}
                        </div>
                    </div>
                ` : ''}
                
                <button class="add-course-btn" data-course-id="${course.course_id}"
                        ${course.is_blocked || (course.missing_prerequisites && course.missing_prerequisites.length > 0) ? 'disabled' : ''}>
                    ${course.is_blocked || (course.missing_prerequisites && course.missing_prerequisites.length > 0) ? 'Blocked' : 'Add Course'}
                </button>
            `;
            container.appendChild(courseCard);
        });

        this.filterCourses();
    }

    getDifficultyColor(difficulty) {
        const difficultyColors = {
            'Easy': '#48bb78',
            'Medium': '#ed8936',
            'Hard': '#e53e3e',
            'Low': '#48bb78',
            'Moderate': '#ed8936',
            'High': '#e53e3e'
        };
        return difficultyColors[difficulty] || '#718096';
    }

    getCompatibilityColor(score) {
        if (score >= 85) return '#48bb78';
        if (score >= 70) return '#ed8936';
        return '#e53e3e';
    }

    filterCourses() {
        const showPrereqOnly = document.getElementById('showPrereqOnly')?.checked;
        const respectCourseLoad = document.getElementById('respectCourseLoad')?.checked;
        const matchLearningStyle = document.getElementById('matchLearningStyle')?.checked;

        const courseCards = document.querySelectorAll('.course-card');
        let totalCreditsShown = 0;
        // Interpret preferred course load as credits per semester, not total
        const preferredLoad = (this.studentData?.preferred_course_load || 15) * 3; // Allow up to 3x preferred load

        courseCards.forEach(card => {
            const courseId = card.querySelector('.add-course-btn').getAttribute('data-course-id');
            const course = this.availableCourses.find(c => c.course_id === courseId);

            let shouldShow = true;

            // Filter by prerequisite priority
            if (showPrereqOnly && !course.is_prerequisite) {
                shouldShow = false;
            }

            // Filter by learning style compatibility (lowered threshold)
            if (matchLearningStyle && course.faculty_compatibility && course.faculty_compatibility < 50) {
                shouldShow = false;
            }

            // Filter by course load (more lenient)
            if (respectCourseLoad && (totalCreditsShown + (course.credits || 3)) > preferredLoad) {
                shouldShow = false;
            }

            if (shouldShow && !this.selectedCourses.includes(courseId)) {
                card.style.display = 'block';
                if (respectCourseLoad) {
                    totalCreditsShown += (course.credits || 3);
                }
            } else {
                card.style.display = 'none';
            }
        });
    }

    addCourse(courseId) {
        if (this.selectedCourses.includes(courseId)) return;

        const course = this.availableCourses.find(c => c.course_id === courseId);
        if (!course || course.is_blocked || (course.missing_prerequisites && course.missing_prerequisites.length > 0)) return;

        this.selectedCourses.push(courseId);
        this.updateSelectedCourses();
        this.updateCourseSummary();

        // Hide the course from recommendations
        const courseCard = document.querySelector(`[data-course-id="${courseId}"]`).closest('.course-card');
        if (courseCard) {
            courseCard.style.display = 'none';
        }

        this.showNotification(`Added ${courseId} to your course plan`, 'success');
    }

    removeCourse(courseId) {
        const index = this.selectedCourses.indexOf(courseId);
        if (index > -1) {
            this.selectedCourses.splice(index, 1);
            this.updateSelectedCourses();
            this.updateCourseSummary();
            this.filterCourses(); // Re-show courses in recommendations

            this.showNotification(`Removed ${courseId} from your course plan`, 'info');
        }
    }

    updateSelectedCourses() {
        const container = document.getElementById('selectedList');

        if (this.selectedCourses.length === 0) {
            container.innerHTML = '<p class="empty-state">No courses selected yet</p>';
            return;
        }

        container.innerHTML = this.selectedCourses.map(courseId => {
            const course = this.availableCourses.find(c => c.course_id === courseId);
            return `
                <div class="selected-course-item">
                    <div class="course-info">
                        <strong>${course.course_id}</strong> - ${course.course_name || course.course_id}
                        <span class="credits">${course.credits || 3} credits</span>
                    </div>
                    <button class="remove-course-btn" data-course-id="${courseId}">Remove</button>
                </div>
            `;
        }).join('');
    }

    updateCourseSummary() {
        const totalCredits = this.selectedCourses.reduce((sum, courseId) => {
            const course = this.availableCourses.find(c => c.course_id === courseId);
            return sum + (course ? (course.credits || 3) : 0);
        }, 0);

        const preferredLoad = this.studentData?.preferred_course_load || 15;
        let workloadAssessment = 'Light';
        if (totalCredits >= preferredLoad * 0.8) workloadAssessment = 'Optimal';
        if (totalCredits > preferredLoad) workloadAssessment = 'Heavy';

        // Calculate faculty compatibility
        const avgCompatibility = this.selectedCourses.length > 0 ?
            this.selectedCourses.reduce((sum, courseId) => {
                const course = this.availableCourses.find(c => c.course_id === courseId);
                return sum + (course ? (course.faculty_compatibility || 0) : 0);
            }, 0) / this.selectedCourses.length : 0;

        document.getElementById('totalCredits').textContent = totalCredits;
        document.getElementById('workloadAssessment').textContent = workloadAssessment;
        document.getElementById('facultyCompatibility').textContent =
            avgCompatibility > 0 ? `${Math.round(avgCompatibility)}%` : '-';

        // Enable/disable export button
        const exportBtn = document.getElementById('exportPlanBtn');
        if (exportBtn) {
            exportBtn.disabled = this.selectedCourses.length === 0;
        }
    }

    exportCoursePlan() {
        if (this.selectedCourses.length === 0) return;

        const plan = {
            student_id: this.studentId,
            term: this.selectedTerm,
            courses: this.selectedCourses.map(courseId => {
                const course = this.availableCourses.find(c => c.course_id === courseId);
                return {
                    course_id: courseId,
                    course_name: course.course_name || courseId,
                    credits: course.credits || 3,
                    faculty: course.recommended_faculty || 'TBD'
                };
            }),
            total_credits: this.selectedCourses.reduce((sum, courseId) => {
                const course = this.availableCourses.find(c => c.course_id === courseId);
                return sum + (course ? (course.credits || 3) : 0);
            }, 0),
            generated_at: new Date().toISOString()
        };

        // Download as JSON file
        const blob = new Blob([JSON.stringify(plan, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `course_plan_${this.studentId}_${this.selectedTerm.id}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        this.showNotification('Course plan exported successfully!', 'success');
    }

    showLoading(message) {
        const overlay = document.getElementById('loadingOverlay');
        const messageElement = overlay.querySelector('p');
        if (messageElement) {
            messageElement.textContent = message;
        }
        overlay.classList.remove('hidden');
    }

    hideLoading() {
        document.getElementById('loadingOverlay').classList.add('hidden');
    }

    showNotification(message, type = 'info') {
        const notification = document.getElementById('notification');
        const textElement = document.getElementById('notificationText');

        textElement.textContent = message;
        notification.className = `notification ${type}`;
        notification.classList.remove('hidden');

        // Auto hide after 3 seconds
        setTimeout(() => {
            notification.classList.add('hidden');
        }, 3000);
    }
}

function hideNotification() {
    document.getElementById('notification').classList.add('hidden');
}

// Initialize application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new CoursePlanning();
});

// Global function for navigation
function goToStudentOptions() {
    const urlParams = new URLSearchParams(window.location.search);
    const studentId = urlParams.get('studentId');
    
    if (studentId) {
        window.location.href = `/student-options?studentId=${encodeURIComponent(studentId)}`;
    } else {
        window.location.href = '/student-options';
    }
}