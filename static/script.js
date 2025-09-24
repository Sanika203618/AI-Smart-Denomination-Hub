// Hero heading animation
anime({
    targets: '.hero h1',
    translateY: [-50, 0],
    opacity: [0, 1],
    duration: 2000,
    easing: 'easeOutElastic(1, .8)'
  });
  
  // Intro paragraph fade-in
  anime({
    targets: '.intro',
    opacity: [0, 1],
    translateY: [30, 0],
    delay: 800,
    duration: 1500,
    easing: 'easeOutQuad'
  });
  
  // Features fade & slide in one by one
  anime({
    targets: '.feature',
    opacity: [0, 1],
    translateY: [40, 0],
    delay: anime.stagger(400, { start: 1500 }),
    duration: 1200,
    easing: 'easeOutBack'
  });
  
  // CTA button glow animation
  anime({
    targets: '.start-btn',
    scale: [0.8, 1],
    opacity: [0, 1],
    delay: 3500,
    duration: 1500,
    easing: 'easeOutElastic(1, .8)',
    loop: false
  });
  