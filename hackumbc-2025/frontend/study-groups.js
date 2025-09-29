// Study Groups JavaScript
class StudyGroupsManager {
    constructor() {
        this.apiBase = window.location.origin;
        this.courses = [];
        this.students = [];
        this.currentPartners = [];
        this.currentGroups = [];
        this.currentStudentId = this.getStudentIdFromURL();
        
        this.init();
    }
    
    getStudentIdFromURL() {
        const urlParams = new URLSearchParams(window.location.search);
        return urlParams.get('studentId');
    }
    
    async init() {
        await this.loadCourses();
        await this.loadStudents();
        this.setupEventListeners();
        this.prePopulateStudentId();
    }
    
    prePopulateStudentId() {
        if (this.currentStudentId) {
            const studentIdField = document.getElementById('partner-student-id');
            if (studentIdField) {
                studentIdField.value = this.currentStudentId;
            }
        }
    }
    
    goToStudentOptions() {
        if (this.currentStudentId) {
            window.location.href = `/student-options?studentId=${encodeURIComponent(this.currentStudentId)}`;
        } else {
            window.location.href = '/student-options';
        }
    }
    
    async loadCourses() {
        try {
            const response = await fetch(`${this.apiBase}/courses`);
            if (response.ok) {
                const data = await response.json();
                this.courses = data.courses || [];
                this.populateCourseDropdowns();
            } else {
                console.error('Failed to load courses');
                this.showError('Failed to load courses. Please try again.');
            }
        } catch (error) {
            console.error('Error loading courses:', error);
            this.showError('Error connecting to the server. Please check your connection.');
        }
    }
    
    async loadStudents() {
        try {
            const response = await fetch(`${this.apiBase}/students`);
            if (response.ok) {
                const data = await response.json();
                this.students = data.students || [];
            } else {
                console.error('Failed to load students');
            }
        } catch (error) {
            console.error('Error loading students:', error);
        }
    }
    
    populateCourseDropdowns() {
        const partnerCourseSelect = document.getElementById('partner-course-id');
        const groupsCourseSelect = document.getElementById('groups-course-id');
        
        // Clear existing options
        partnerCourseSelect.innerHTML = '<option value="">Select a course</option>';
        groupsCourseSelect.innerHTML = '<option value="">Select a course</option>';
        
        // Add course options
        this.courses.forEach(courseId => {
            const option1 = document.createElement('option');
            option1.value = courseId;
            option1.textContent = courseId;
            partnerCourseSelect.appendChild(option1);
            
            const option2 = document.createElement('option');
            option2.value = courseId;
            option2.textContent = courseId;
            groupsCourseSelect.appendChild(option2);
        });
    }
    
    setupEventListeners() {
        // Modal close buttons
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('modal')) {
                this.closeModals();
            }
        });
        
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.closeModals();
            }
        });
    }
    
    async findStudyPartners() {
        const studentId = document.getElementById('partner-student-id').value.trim();
        const courseId = document.getElementById('partner-course-id').value;
        const maxPartners = parseInt(document.getElementById('max-partners').value) || 10;
        
        if (!studentId || !courseId) {
            this.showError('Please enter your Student ID and select a course.');
            return;
        }
        
        this.showLoading();
        this.hideResults();
        
        try {
            const response = await fetch(`${this.apiBase}/study-partners`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    student_id: studentId,
                    course_id: courseId,
                    max_partners: maxPartners
                })
            });
            
            if (response.ok) {
                const partners = await response.json();
                this.currentPartners = partners;
                this.displayPartners(partners);
            } else {
                const errorData = await response.json();
                this.showError(errorData.detail || 'Failed to find study partners.');
            }
        } catch (error) {
            console.error('Error finding study partners:', error);
            this.showError('Error connecting to the server. Please try again.');
        } finally {
            this.hideLoading();
        }
    }
    
    async createStudyGroups() {
        const courseId = document.getElementById('groups-course-id').value;
        const minGroupSize = parseInt(document.getElementById('min-group-size').value) || 3;
        const maxGroupSize = parseInt(document.getElementById('max-group-size').value) || 5;
        
        if (!courseId) {
            this.showError('Please select a course.');
            return;
        }
        
        if (minGroupSize > maxGroupSize) {
            this.showError('Minimum group size cannot be larger than maximum group size.');
            return;
        }
        
        this.showLoading();
        this.hideResults();
        
        try {
            const response = await fetch(`${this.apiBase}/study-groups`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    course_id: courseId,
                    min_group_size: minGroupSize,
                    max_group_size: maxGroupSize
                })
            });
            
            if (response.ok) {
                const groups = await response.json();
                this.currentGroups = groups;
                this.displayGroups(groups);
            } else {
                const errorData = await response.json();
                this.showError(errorData.detail || 'Failed to create study groups.');
            }
        } catch (error) {
            console.error('Error creating study groups:', error);
            this.showError('Error connecting to the server. Please try again.');
        } finally {
            this.hideLoading();
        }
    }
    
    displayPartners(partners) {
        const partnersResults = document.getElementById('partners-results');
        const partnersList = document.getElementById('partners-list');
        const partnersCount = document.getElementById('partners-count');
        
        partnersCount.textContent = partners.length;
        
        if (partners.length === 0) {
            partnersList.innerHTML = `
                <div style="grid-column: 1 / -1; text-align: center; padding: 2rem; color: #64748b;">
                    <i class="fas fa-search" style="font-size: 3rem; margin-bottom: 1rem; color: #cbd5e1;"></i>
                    <h3>No compatible study partners found</h3>
                    <p>Try adjusting your search criteria or check back later as more students join courses.</p>
                </div>
            `;
        } else {
            partnersList.innerHTML = partners.map(partner => this.createPartnerCard(partner)).join('');
        }
        
        partnersResults.style.display = 'block';
        partnersResults.scrollIntoView({ behavior: 'smooth' });
    }
    
    displayGroups(groups) {
        const groupsResults = document.getElementById('groups-results');
        const groupsList = document.getElementById('groups-list');
        const groupsCount = document.getElementById('groups-count');
        
        groupsCount.textContent = groups.length;
        
        if (groups.length === 0) {
            groupsList.innerHTML = `
                <div style="text-align: center; padding: 2rem; color: #64748b;">
                    <i class="fas fa-users" style="font-size: 3rem; margin-bottom: 1rem; color: #cbd5e1;"></i>
                    <h3>No study groups could be formed</h3>
                    <p>There may not be enough students enrolled in this course or the group size requirements are too restrictive.</p>
                </div>
            `;
        } else {
            groupsList.innerHTML = groups.map(group => this.createGroupCard(group)).join('');
        }
        
        groupsResults.style.display = 'block';
        groupsResults.scrollIntoView({ behavior: 'smooth' });
    }
    
    createPartnerCard(partner) {
        const compatibilityClass = this.getCompatibilityClass(partner.compatibility_score);
        const compatibilityPercentage = Math.round(partner.compatibility_score * 100);
        
        return `
            <div class="partner-card" onclick="studyGroupsManager.showPartnerDetails('${partner.student_id}')">
                <div class="partner-header">
                    <div>
                        <h3 class="partner-name">${this.escapeHtml(partner.name)}</h3>
                        <div class="partner-id">${this.escapeHtml(partner.student_id)}</div>
                    </div>
                    <div class="compatibility-score">
                        <div class="score-circle ${compatibilityClass}">
                            ${compatibilityPercentage}%
                        </div>
                        <div class="score-label">Match</div>
                    </div>
                </div>
                
                <div class="partner-details">
                    <div class="detail-item">
                        <i class="fas fa-brain"></i>
                        <span>${this.escapeHtml(partner.learning_style)}</span>
                    </div>
                    <div class="detail-item">
                        <i class="fas fa-chart-line"></i>
                        <span>${this.escapeHtml(partner.performance_level)}</span>
                    </div>
                    <div class="detail-item">
                        <i class="fas fa-clock"></i>
                        <span>${partner.work_hours}h/week</span>
                    </div>
                    <div class="detail-item">
                        <i class="fas fa-graduation-cap"></i>
                        <span>${this.escapeHtml(partner.preferred_pace)}</span>
                    </div>
                </div>
                
                <div class="partner-tags">
                    <span class="tag learning-style">${this.escapeHtml(partner.learning_style)} Learner</span>
                    <span class="tag performance">${this.escapeHtml(partner.performance_level)} Performance</span>
                    <span class="tag">${this.escapeHtml(partner.instruction_mode)}</span>
                </div>
            </div>
        `;
    }
    
    createGroupCard(group) {
        const compatibilityPercentage = Math.round(group.avg_compatibility * 100);
        const diversityPercentage = Math.round(group.learning_style_diversity * 100);
        const balancePercentage = Math.round(group.performance_balance * 100);
        
        return `
            <div class="group-card" onclick="studyGroupsManager.showGroupDetails('${group.group_id}')">
                <div class="group-header">
                    <div class="group-title">
                        <h3 class="group-name">${this.escapeHtml(group.group_id)}</h3>
                        <span class="group-size">${group.group_size} members</span>
                    </div>
                    <div class="group-course">${this.escapeHtml(group.course_name)}</div>
                </div>
                
                <div class="group-stats">
                    <div class="stat-item">
                        <span class="stat-value">${compatibilityPercentage}%</span>
                        <div class="stat-label">Compatibility</div>
                    </div>
                    <div class="stat-item">
                        <span class="stat-value">${diversityPercentage}%</span>
                        <div class="stat-label">Learning Diversity</div>
                    </div>
                    <div class="stat-item">
                        <span class="stat-value">${balancePercentage}%</span>
                        <div class="stat-label">Performance Balance</div>
                    </div>
                </div>
                
                <div class="group-meeting">
                    <div class="meeting-icon">
                        <i class="fas fa-calendar-alt"></i>
                    </div>
                    <div class="meeting-time">${this.escapeHtml(group.recommended_meeting_time)}</div>
                    <div class="meeting-label">Recommended Meeting Time</div>
                </div>
                
                <div class="group-members">
                    <div class="members-header">
                        <i class="fas fa-users"></i>
                        Group Members
                    </div>
                    <div class="members-list">
                        ${group.members.slice(0, 3).map(member => `
                            <div class="member-item">
                                <div class="member-name">${this.escapeHtml(member.name)}</div>
                                <div class="member-details">
                                    ${this.escapeHtml(member.learning_style)} • ${this.escapeHtml(member.performance_level)} Performance
                                </div>
                            </div>
                        `).join('')}
                        ${group.members.length > 3 ? `
                            <div class="member-item" style="display: flex; align-items: center; justify-content: center; font-weight: 600; color: #667eea;">
                                +${group.members.length - 3} more
                            </div>
                        ` : ''}
                    </div>
                </div>
            </div>
        `;
    }
    
    showPartnerDetails(partnerId) {
        const partner = this.currentPartners.find(p => p.student_id === partnerId);
        if (!partner) return;
        
        const modal = document.getElementById('partner-modal');
        const detailsContainer = document.getElementById('partner-details');
        
        const compatibilityFactors = Object.entries(partner.compatibility_factors)
            .map(([factor, score]) => `
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.5rem 0; border-bottom: 1px solid #e2e8f0;">
                    <span style="text-transform: capitalize;">${factor.replace('_', ' ')}</span>
                    <span style="font-weight: 600; color: ${this.getScoreColor(score)};">
                        ${Math.round(score * 100)}%
                    </span>
                </div>
            `).join('');
        
        detailsContainer.innerHTML = `
            <div style="text-align: center; margin-bottom: 2rem;">
                <h2 style="margin: 0; color: #1e293b;">${this.escapeHtml(partner.name)}</h2>
                <p style="color: #64748b; margin: 0.5rem 0;">${this.escapeHtml(partner.student_id)}</p>
                <div style="display: inline-block; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                           color: white; padding: 0.5rem 1rem; border-radius: 8px; font-weight: 600;">
                    ${Math.round(partner.compatibility_score * 100)}% Compatibility
                </div>
            </div>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; margin-bottom: 2rem;">
                <div>
                    <h4 style="color: #1e293b; margin-bottom: 1rem;">Personal Details</h4>
                    <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                        <i class="fas fa-brain" style="color: #667eea; width: 20px;"></i>
                        <span><strong>Learning Style:</strong> ${this.escapeHtml(partner.learning_style)}</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                        <i class="fas fa-chart-line" style="color: #667eea; width: 20px;"></i>
                        <span><strong>Performance:</strong> ${this.escapeHtml(partner.performance_level)}</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                        <i class="fas fa-clock" style="color: #667eea; width: 20px;"></i>
                        <span><strong>Work Hours:</strong> ${partner.work_hours}/week</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                        <i class="fas fa-tachometer-alt" style="color: #667eea; width: 20px;"></i>
                        <span><strong>Preferred Pace:</strong> ${this.escapeHtml(partner.preferred_pace)}</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                        <i class="fas fa-laptop" style="color: #667eea; width: 20px;"></i>
                        <span><strong>Instruction Mode:</strong> ${this.escapeHtml(partner.instruction_mode)}</span>
                    </div>
                </div>
                
                <div>
                    <h4 style="color: #1e293b; margin-bottom: 1rem;">Course Information</h4>
                    <div style="margin-bottom: 1rem;">
                        <strong>Current Courses (${partner.current_courses.length}):</strong>
                        <div style="margin-top: 0.5rem;">
                            ${partner.current_courses.slice(0, 5).map(course => 
                                `<span style="background: #eff6ff; color: #1d4ed8; padding: 0.25rem 0.5rem; 
                                 border-radius: 4px; font-size: 0.8rem; margin: 0.25rem 0.25rem 0.25rem 0; 
                                 display: inline-block;">${this.escapeHtml(course)}</span>`
                            ).join('')}
                            ${partner.current_courses.length > 5 ? 
                                `<span style="color: #64748b; font-size: 0.9rem;">+${partner.current_courses.length - 5} more</span>` 
                                : ''}
                        </div>
                    </div>
                    <div>
                        <strong>Completed Courses (${partner.completed_courses.length}):</strong>
                        <div style="margin-top: 0.5rem;">
                            ${partner.completed_courses.slice(0, 5).map(course => 
                                `<span style="background: #f0fdf4; color: #16a34a; padding: 0.25rem 0.5rem; 
                                 border-radius: 4px; font-size: 0.8rem; margin: 0.25rem 0.25rem 0.25rem 0; 
                                 display: inline-block;">${this.escapeHtml(course)}</span>`
                            ).join('')}
                            ${partner.completed_courses.length > 5 ? 
                                `<span style="color: #64748b; font-size: 0.9rem;">+${partner.completed_courses.length - 5} more</span>` 
                                : ''}
                        </div>
                    </div>
                </div>
            </div>
            
            <div>
                <h4 style="color: #1e293b; margin-bottom: 1rem;">Compatibility Breakdown</h4>
                <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1rem;">
                    ${compatibilityFactors}
                </div>
            </div>
        `;
        
        modal.style.display = 'flex';
    }
    
    showGroupDetails(groupId) {
        const group = this.currentGroups.find(g => g.group_id === groupId);
        if (!group) return;
        
        const modal = document.getElementById('group-modal');
        const detailsContainer = document.getElementById('group-details');
        
        detailsContainer.innerHTML = `
            <div style="text-align: center; margin-bottom: 2rem;">
                <h2 style="margin: 0; color: #1e293b;">${this.escapeHtml(group.group_id)}</h2>
                <p style="color: #64748b; margin: 0.5rem 0;">${this.escapeHtml(group.course_name)}</p>
                <div style="display: flex; gap: 1rem; justify-content: center; margin-top: 1rem;">
                    <div style="background: #eff6ff; color: #1d4ed8; padding: 0.5rem 1rem; border-radius: 8px; font-weight: 600;">
                        ${Math.round(group.avg_compatibility * 100)}% Compatibility
                    </div>
                    <div style="background: #f0fdf4; color: #16a34a; padding: 0.5rem 1rem; border-radius: 8px; font-weight: 600;">
                        ${group.group_size} Members
                    </div>
                </div>
            </div>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; margin-bottom: 2rem;">
                <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1rem;">
                    <h4 style="color: #1e293b; margin-bottom: 1rem;">Group Statistics</h4>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                        <span>Average Compatibility:</span>
                        <span style="font-weight: 600;">${Math.round(group.avg_compatibility * 100)}%</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                        <span>Learning Style Diversity:</span>
                        <span style="font-weight: 600;">${Math.round(group.learning_style_diversity * 100)}%</span>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span>Performance Balance:</span>
                        <span style="font-weight: 600;">${Math.round(group.performance_balance * 100)}%</span>
                    </div>
                </div>
                
                <div style="background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 8px; padding: 1rem; text-align: center;">
                    <h4 style="color: #1e40af; margin-bottom: 1rem;">
                        <i class="fas fa-calendar-alt"></i> Recommended Meeting
                    </h4>
                    <div style="font-size: 1.1rem; font-weight: 600; color: #1e40af;">
                        ${this.escapeHtml(group.recommended_meeting_time)}
                    </div>
                </div>
            </div>
            
            <div>
                <h4 style="color: #1e293b; margin-bottom: 1rem;">Group Members</h4>
                <div style="display: grid; gap: 1rem;">
                    ${group.members.map(member => `
                        <div style="background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1rem; 
                                   display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; align-items: center;">
                            <div>
                                <div style="font-weight: 600; color: #1e293b; margin-bottom: 0.25rem;">
                                    ${this.escapeHtml(member.name)}
                                </div>
                                <div style="color: #64748b; font-size: 0.9rem;">
                                    ${this.escapeHtml(member.student_id)}
                                </div>
                            </div>
                            <div style="display: flex; flex-wrap: wrap; gap: 0.5rem; justify-content: flex-end;">
                                <span style="background: #ede9fe; color: #7c3aed; padding: 0.25rem 0.5rem; 
                                           border-radius: 4px; font-size: 0.8rem;">
                                    ${this.escapeHtml(member.learning_style)}
                                </span>
                                <span style="background: #dcfce7; color: #16a34a; padding: 0.25rem 0.5rem; 
                                           border-radius: 4px; font-size: 0.8rem;">
                                    ${this.escapeHtml(member.performance_level)}
                                </span>
                                <span style="background: #f1f5f9; color: #475569; padding: 0.25rem 0.5rem; 
                                           border-radius: 4px; font-size: 0.8rem;">
                                    ${member.work_hours}h/week
                                </span>
                            </div>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
        
        modal.style.display = 'flex';
    }
    
    closePartnerModal() {
        document.getElementById('partner-modal').style.display = 'none';
    }
    
    closeGroupModal() {
        document.getElementById('group-modal').style.display = 'none';
    }
    
    closeModals() {
        this.closePartnerModal();
        this.closeGroupModal();
    }
    
    showLoading() {
        document.getElementById('loading').style.display = 'block';
    }
    
    hideLoading() {
        document.getElementById('loading').style.display = 'none';
    }
    
    hideResults() {
        document.getElementById('partners-results').style.display = 'none';
        document.getElementById('groups-results').style.display = 'none';
        document.getElementById('error-message').style.display = 'none';
    }
    
    showError(message) {
        const errorContainer = document.getElementById('error-message');
        const errorText = document.getElementById('error-text');
        
        errorText.textContent = message;
        errorContainer.style.display = 'block';
        errorContainer.scrollIntoView({ behavior: 'smooth' });
    }
    
    getCompatibilityClass(score) {
        if (score >= 0.7) return 'score-high';
        if (score >= 0.5) return 'score-medium';
        return 'score-low';
    }
    
    getScoreColor(score) {
        if (score >= 0.7) return '#16a34a';
        if (score >= 0.5) return '#d97706';
        return '#dc2626';
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Global functions for onclick handlers
function findStudyPartners() {
    console.log('findStudyPartners called');
    console.log('studyGroupsManager:', studyGroupsManager);
    if (studyGroupsManager) {
        studyGroupsManager.findStudyPartners();
    } else {
        console.error('studyGroupsManager not initialized');
        alert('Study Groups Manager not initialized. Please refresh the page.');
    }
}

function createStudyGroups() {
    studyGroupsManager.createStudyGroups();
}

function closePartnerModal() {
    studyGroupsManager.closePartnerModal();
}

function closeGroupModal() {
    studyGroupsManager.closeGroupModal();
}

// Initialize when DOM is loaded
let studyGroupsManager;

document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM loaded, initializing StudyGroupsManager');
    try {
        studyGroupsManager = new StudyGroupsManager();
        console.log('StudyGroupsManager initialized successfully:', studyGroupsManager);
    } catch (error) {
        console.error('Error initializing StudyGroupsManager:', error);
        alert('Error initializing Study Groups Manager: ' + error.message);
    }
});

// Global function for navigation
function goToStudentOptions() {
    if (studyGroupsManager) {
        studyGroupsManager.goToStudentOptions();
    } else {
        window.location.href = '/student-options';
    }
}
