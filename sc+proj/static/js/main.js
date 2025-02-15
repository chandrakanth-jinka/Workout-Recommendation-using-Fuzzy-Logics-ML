document.addEventListener('DOMContentLoaded', function () {
    // Sidebar toggle
    document.querySelector('.nav-toggle').addEventListener('click', function () {
        document.querySelector('.sidebar').classList.toggle('active');
        document.querySelector('.container').classList.toggle('shifted');
    });

    // Theme toggle functionality
    const themeToggle = document.querySelector('.theme-toggle');
    const root = document.documentElement;

    // Check for saved theme preference, default to dark if none saved
    const savedTheme = localStorage.getItem('theme') || 'dark';
    setTheme(savedTheme);

    themeToggle.addEventListener('click', () => {
        const currentTheme = root.getAttribute('data-theme');
        const newTheme = currentTheme === 'light' ? 'dark' : 'light';
        setTheme(newTheme);
        localStorage.setItem('theme', newTheme);
    });

    function setTheme(theme) {
        root.setAttribute('data-theme', theme);
        themeToggle.setAttribute('data-theme', theme);
    }

    // Add active class to current nav item
    const currentPath = window.location.pathname;
    document.querySelectorAll('.nav-item').forEach(item => {
        if (item.getAttribute('href') === currentPath) {
            item.classList.add('active');
        }
    });

    // Form loading state
    const form = document.querySelector('form');
    if (form) {
        form.addEventListener('submit', function () {
            const button = this.querySelector('button');
            button.classList.add('loading');
            button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
        });
    }
}); 