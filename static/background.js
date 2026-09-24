/**
 * Mol_Smell - Interactive 3D Background Canvas Engine
 * Features:
 * - Rotating 3D DNA Double Helix with glowing nucleotide base pairs
 * - Authentic 3D fragrance molecules (Benzene, Vanillin, Limonene, Ester) with bonds & atoms
 * - Liquid-wobbling scent droplets/vesicles with organic harmonic deformation
 * - Ambient olfactory sensory dust field
 * - Interactive mouse parallax with smooth damping
 * - High-DPI crisp rendering & 60fps performance
 */

(function () {
    'use strict';

    function init() {
        let canvas = document.getElementById('bg-canvas');
        if (!canvas) {
            canvas = document.createElement('canvas');
            canvas.id = 'bg-canvas';
            canvas.setAttribute('aria-hidden', 'true');
            if (document.body) {
                document.body.prepend(canvas);
            } else {
                window.addEventListener('DOMContentLoaded', () => document.body.prepend(canvas));
            }
        }

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        let width = 0;
        let height = 0;
        let dpr = 1;

        // Mouse tracking for smooth parallax
        const mouse = { x: 0, y: 0, targetX: 0, targetY: 0 };
        window.addEventListener('mousemove', (e) => {
            mouse.targetX = (e.clientX / window.innerWidth) * 2 - 1;
            mouse.targetY = (e.clientY / window.innerHeight) * 2 - 1;
        });

        // Handle touch move for mobile devices
        window.addEventListener('touchmove', (e) => {
            if (e.touches.length > 0) {
                mouse.targetX = (e.touches[0].clientX / window.innerWidth) * 2 - 1;
                mouse.targetY = (e.touches[0].clientY / window.innerHeight) * 2 - 1;
            }
        }, { passive: true });

        function resize() {
            dpr = Math.min(window.devicePixelRatio || 1, 2);
            width = window.innerWidth;
            height = window.innerHeight;
            canvas.width = width * dpr;
            canvas.height = height * dpr;
            canvas.style.width = width + 'px';
            canvas.style.height = height + 'px';
            ctx.scale(dpr, dpr);
        }
        window.addEventListener('resize', resize);
        resize();

        // 3D Math Utilities
        const FOV = 450;

        function project(x, y, z, cx, cy) {
            const perspective = FOV / (FOV + z);
            return {
                x: cx + x * perspective,
                y: cy + y * perspective,
                scale: perspective,
                z: z
            };
        }

        function rotateX(p, angle) {
            const cos = Math.cos(angle);
            const sin = Math.sin(angle);
            return {
                x: p.x,
                y: p.y * cos - p.z * sin,
                z: p.y * sin + p.z * cos
            };
        }

        function rotateY(p, angle) {
            const cos = Math.cos(angle);
            const sin = Math.sin(angle);
            return {
                x: p.x * cos + p.z * sin,
                y: p.y,
                z: -p.x * sin + p.z * cos
            };
        }

        function rotateZ(p, angle) {
            const cos = Math.cos(angle);
            const sin = Math.sin(angle);
            return {
                x: p.x * cos - p.y * sin,
                y: p.x * sin + p.y * cos,
                z: p.z
            };
        }

        // -------------------------------------------------------------
        // 1. 3D DNA DOUBLE HELIX
        // -------------------------------------------------------------
        class DNAHelix {
            constructor() {
                this.numNodes = 36;
                this.radius = 65;
                this.length = 900;
                this.twistRate = 0.18;
                this.rotationSpeed = 0.012;
                this.angle = 0;
                this.wobblePhase = Math.random() * Math.PI * 2;
            }

            update(time) {
                this.angle += this.rotationSpeed;
            }

            draw(ctx, time, mouseOffset) {
                // Anchor DNA along diagonal/right side of screen
                const cx = width * 0.82 + mouseOffset.x * 35;
                const cy = height * 0.48 + mouseOffset.y * 25;
                const tilt = -0.32; // diagonal slant

                const items = [];
                const stepY = this.length / this.numNodes;
                const startY = -this.length / 2;

                for (let i = 0; i < this.numNodes; i++) {
                    const py = startY + i * stepY;
                    const nodeAngle = this.angle + i * this.twistRate;

                    // Organic wave wobble running through DNA
                    const wobble = Math.sin(time * 1.5 + i * 0.18 + this.wobblePhase) * 16;
                    const wobbleZ = Math.cos(time * 1.2 + i * 0.15) * 14;

                    // Strand 1 node
                    const p1 = {
                        x: Math.cos(nodeAngle) * this.radius + wobble,
                        y: py,
                        z: Math.sin(nodeAngle) * this.radius + wobbleZ
                    };

                    // Strand 2 node (opposite by PI)
                    const p2 = {
                        x: Math.cos(nodeAngle + Math.PI) * this.radius + wobble,
                        y: py,
                        z: Math.sin(nodeAngle + Math.PI) * this.radius + wobbleZ
                    };

                    // Apply tilt rotation
                    const rotP1 = rotateZ(rotateX(p1, 0.25), tilt);
                    const rotP2 = rotateZ(rotateX(p2, 0.25), tilt);

                    // Add parallax rotation
                    const finalP1 = rotateY(rotP1, mouseOffset.x * 0.2);
                    const finalP2 = rotateY(rotP2, mouseOffset.x * 0.2);

                    const proj1 = project(finalP1.x, finalP1.y, finalP1.z + 120, cx, cy);
                    const proj2 = project(finalP2.x, finalP2.y, finalP2.z + 120, cx, cy);

                    // Base pair type colors: Adenine-Thymine (cyan/gold), Guanine-Cytosine (magenta/emerald)
                    const pairType = i % 4;
                    let color1, color2;
                    if (pairType === 0) {
                        color1 = '#00f2fe'; color2 = '#f6d365';
                    } else if (pairType === 1) {
                        color1 = '#f093fb'; color2 = '#00f5a0';
                    } else if (pairType === 2) {
                        color1 = '#4facfe'; color2 = '#ff758c';
                    } else {
                        color1 = '#a8edea'; color2 = '#fed6e3';
                    }

                    // Push rung (connector)
                    const avgZ = (proj1.z + proj2.z) / 2;
                    items.push({
                        type: 'dna_rung',
                        z: avgZ,
                        p1: proj1,
                        p2: proj2,
                        color1: color1,
                        color2: color2,
                        scale: (proj1.scale + proj2.scale) / 2
                    });

                    // Push strand nodes
                    items.push({
                        type: 'dna_node',
                        z: proj1.z,
                        p: proj1,
                        color: color1,
                        radius: 4.5 * proj1.scale
                    });

                    items.push({
                        type: 'dna_node',
                        z: proj2.z,
                        p: proj2,
                        color: color2,
                        radius: 4.5 * proj2.scale
                    });
                }

                // Draw rungs and nodes sorted by Z depth
                items.sort((a, b) => a.z - b.z);

                for (const item of items) {
                    if (item.type === 'dna_rung') {
                        const grad = ctx.createLinearGradient(item.p1.x, item.p1.y, item.p2.x, item.p2.y);
                        grad.addColorStop(0, item.color1);
                        grad.addColorStop(0.5, 'rgba(255, 255, 255, 0.6)');
                        grad.addColorStop(1, item.color2);

                        const alpha = Math.max(0.12, Math.min(0.55, (item.z + 100) / 200));
                        ctx.save();
                        ctx.globalAlpha = alpha;
                        ctx.strokeStyle = grad;
                        ctx.lineWidth = Math.max(1, 2.2 * item.scale);
                        ctx.beginPath();
                        ctx.moveTo(item.p1.x, item.p1.y);
                        ctx.lineTo(item.p2.x, item.p2.y);
                        ctx.stroke();
                        ctx.restore();
                    } else if (item.type === 'dna_node') {
                        const alpha = Math.max(0.2, Math.min(0.9, (item.z + 100) / 180));
                        ctx.save();
                        ctx.globalAlpha = alpha;

                        const glowRadius = Math.max(1, item.radius * 2.2);
                        const glow = ctx.createRadialGradient(item.p.x, item.p.y, 0, item.p.x, item.p.y, glowRadius);
                        glow.addColorStop(0, item.color);
                        glow.addColorStop(0.5, item.color);
                        glow.addColorStop(1, 'rgba(0,0,0,0)');

                        ctx.fillStyle = glow;
                        ctx.beginPath();
                        ctx.arc(item.p.x, item.p.y, glowRadius, 0, Math.PI * 2);
                        ctx.fill();

                        ctx.fillStyle = '#ffffff';
                        ctx.beginPath();
                        ctx.arc(item.p.x, item.p.y, Math.max(1, item.radius * 0.65), 0, Math.PI * 2);
                        ctx.fill();
                        ctx.restore();
                    }
                }
            }
        }

        // -------------------------------------------------------------
        // 2. 3D FLOATING & WOBBLING MOLECULAR STRUCTURES
        // -------------------------------------------------------------
        const ATOM_COLORS = {
            C: { fill: '#00f2fe', glow: 'rgba(0, 242, 254, 0.4)', radius: 5.5 },
            O: { fill: '#ff4b72', glow: 'rgba(255, 75, 114, 0.4)', radius: 5.8 },
            N: { fill: '#9b51e0', glow: 'rgba(155, 81, 224, 0.4)', radius: 5.5 },
            H: { fill: '#e0f7fa', glow: 'rgba(224, 247, 250, 0.3)', radius: 3.5 }
        };

        const MOLECULE_TEMPLATES = [
            // Benzene Ring
            {
                name: "Benzene",
                atoms: [
                    { type: 'C', x: 0, y: -45, z: 0 },
                    { type: 'C', x: 39, y: -22.5, z: 0 },
                    { type: 'C', x: 39, y: 22.5, z: 0 },
                    { type: 'C', x: 0, y: 45, z: 0 },
                    { type: 'C', x: -39, y: 22.5, z: 0 },
                    { type: 'C', x: -39, y: -22.5, z: 0 },
                    { type: 'H', x: 0, y: -70, z: 0 },
                    { type: 'H', x: 62, y: -35, z: 0 },
                    { type: 'H', x: 62, y: 35, z: 0 },
                    { type: 'H', x: 0, y: 70, z: 0 },
                    { type: 'H', x: -62, y: 35, z: 0 },
                    { type: 'H', x: -62, y: -35, z: 0 }
                ],
                bonds: [
                    { a: 0, b: 1, order: 2 },
                    { a: 1, b: 2, order: 1 },
                    { a: 2, b: 3, order: 2 },
                    { a: 3, b: 4, order: 1 },
                    { a: 4, b: 5, order: 2 },
                    { a: 5, b: 0, order: 1 },
                    { a: 0, b: 6, order: 1 },
                    { a: 1, b: 7, order: 1 },
                    { a: 2, b: 8, order: 1 },
                    { a: 3, b: 9, order: 1 },
                    { a: 4, b: 10, order: 1 },
                    { a: 5, b: 11, order: 1 }
                ]
            },
            // Vanillin
            {
                name: "Vanillin",
                atoms: [
                    { type: 'C', x: -20, y: -35, z: 0 },
                    { type: 'C', x: 20, y: -35, z: 0 },
                    { type: 'C', x: 40, y: 0, z: 0 },
                    { type: 'C', x: 20, y: 35, z: 0 },
                    { type: 'C', x: -20, y: 35, z: 0 },
                    { type: 'C', x: -40, y: 0, z: 0 },
                    { type: 'C', x: -35, y: -65, z: 10 },
                    { type: 'O', x: -35, y: -90, z: 15 },
                    { type: 'H', x: -55, y: -65, z: 10 },
                    { type: 'O', x: 40, y: 65, z: -10 },
                    { type: 'H', x: 60, y: 65, z: -10 },
                    { type: 'O', x: 60, y: 0, z: 15 },
                    { type: 'C', x: 85, y: 0, z: 20 }
                ],
                bonds: [
                    { a: 0, b: 1, order: 2 },
                    { a: 1, b: 2, order: 1 },
                    { a: 2, b: 3, order: 2 },
                    { a: 3, b: 4, order: 1 },
                    { a: 4, b: 5, order: 2 },
                    { a: 5, b: 0, order: 1 },
                    { a: 0, b: 6, order: 1 },
                    { a: 6, b: 7, order: 2 },
                    { a: 6, b: 8, order: 1 },
                    { a: 3, b: 9, order: 1 },
                    { a: 9, b: 10, order: 1 },
                    { a: 2, b: 11, order: 1 },
                    { a: 11, b: 12, order: 1 }
                ]
            },
            // Limonene
            {
                name: "Limonene",
                atoms: [
                    { type: 'C', x: -25, y: -30, z: 10 },
                    { type: 'C', x: 20, y: -30, z: -10 },
                    { type: 'C', x: 40, y: 0, z: 10 },
                    { type: 'C', x: 20, y: 35, z: -10 },
                    { type: 'C', x: -25, y: 35, z: 10 },
                    { type: 'C', x: -45, y: 0, z: -10 },
                    { type: 'C', x: -40, y: -60, z: 15 },
                    { type: 'C', x: 45, y: 65, z: -15 },
                    { type: 'C', x: 70, y: 65, z: -20 },
                    { type: 'C', x: 35, y: 92, z: -15 }
                ],
                bonds: [
                    { a: 0, b: 1, order: 2 },
                    { a: 1, b: 2, order: 1 },
                    { a: 2, b: 3, order: 1 },
                    { a: 3, b: 4, order: 1 },
                    { a: 4, b: 5, order: 1 },
                    { a: 5, b: 0, order: 1 },
                    { a: 0, b: 6, order: 1 },
                    { a: 3, b: 7, order: 1 },
                    { a: 7, b: 8, order: 2 },
                    { a: 7, b: 9, order: 1 }
                ]
            },
            // Ester
            {
                name: "Ester",
                atoms: [
                    { type: 'C', x: -65, y: 15, z: 0 },
                    { type: 'C', x: -35, y: -10, z: 10 },
                    { type: 'C', x: -5, y: 15, z: -10 },
                    { type: 'C', x: 25, y: -10, z: 0 },
                    { type: 'O', x: 25, y: -40, z: 5 },
                    { type: 'O', x: 50, y: 15, z: -5 },
                    { type: 'C', x: 75, y: -5, z: 10 },
                    { type: 'C', x: 105, y: 20, z: -10 }
                ],
                bonds: [
                    { a: 0, b: 1, order: 1 },
                    { a: 1, b: 2, order: 1 },
                    { a: 2, b: 3, order: 1 },
                    { a: 3, b: 4, order: 2 },
                    { a: 3, b: 5, order: 1 },
                    { a: 5, b: 6, order: 1 },
                    { a: 6, b: 7, order: 1 }
                ]
            }
        ];

        class MoleculeInstance {
            constructor(templateIndex, posX, posY, posZ) {
                this.template = MOLECULE_TEMPLATES[templateIndex % MOLECULE_TEMPLATES.length];
                this.x = posX;
                this.y = posY;
                this.z = posZ;

                this.vx = (Math.random() - 0.5) * 0.45;
                this.vy = (Math.random() - 0.5) * 0.35;
                this.vz = (Math.random() - 0.5) * 0.3;

                this.rotX = Math.random() * Math.PI * 2;
                this.rotY = Math.random() * Math.PI * 2;
                this.rotZ = Math.random() * Math.PI * 2;
                this.vRotX = (Math.random() - 0.5) * 0.015;
                this.vRotY = (Math.random() - 0.5) * 0.018;
                this.vRotZ = (Math.random() - 0.5) * 0.012;

                this.wobbleFreq = 1.2 + Math.random() * 1.5;
                this.wobblePhase = Math.random() * Math.PI * 2;
                this.wobbleAmp = 0.07 + Math.random() * 0.06;
                this.baseScale = 0.85 + Math.random() * 0.35;
            }

            update(time) {
                this.x += this.vx;
                this.y += this.vy;
                this.z += this.vz;

                const margin = 180;
                if (this.x < -margin) this.x = width + margin;
                if (this.x > width + margin) this.x = -margin;
                if (this.y < -margin) this.y = height + margin;
                if (this.y > height + margin) this.y = -margin;
                if (this.z < -100) this.z = 250;
                if (this.z > 250) this.z = -100;

                this.rotX += this.vRotX;
                this.rotY += this.vRotY;
                this.rotZ += this.vRotZ;
            }

            draw(ctx, time, mouseOffset) {
                const wobble = Math.sin(time * this.wobbleFreq + this.wobblePhase);
                const wobbleScale = this.baseScale * (1 + wobble * this.wobbleAmp);
                const wobbleRot = Math.cos(time * (this.wobbleFreq * 0.8) + this.wobblePhase) * 0.15;

                const worldX = this.x + mouseOffset.x * (40 + this.z * 0.1);
                const worldY = this.y + mouseOffset.y * (30 + this.z * 0.1);

                const projectedAtoms = this.template.atoms.map((atom) => {
                    let p = {
                        x: atom.x * wobbleScale,
                        y: atom.y * wobbleScale,
                        z: atom.z * wobbleScale
                    };

                    p = rotateX(p, this.rotX + wobbleRot);
                    p = rotateY(p, this.rotY + wobbleRot * 0.7);
                    p = rotateZ(p, this.rotZ);

                    const proj = project(p.x, p.y, p.z + this.z, worldX, worldY);
                    return {
                        ...atom,
                        proj: proj,
                        scale: proj.scale,
                        z: proj.z
                    };
                });

                const alpha = Math.max(0.18, Math.min(0.75, (this.z + 120) / 320));

                ctx.save();
                ctx.globalAlpha = alpha;

                for (const bond of this.template.bonds) {
                    const a1 = projectedAtoms[bond.a];
                    const a2 = projectedAtoms[bond.b];
                    if (!a1 || !a2) continue;

                    const dx = a2.proj.x - a1.proj.x;
                    const dy = a2.proj.y - a1.proj.y;
                    const len = Math.sqrt(dx * dx + dy * dy);
                    if (len < 1) continue;

                    if (bond.order === 2) {
                        const nx = (-dy / len) * (2.8 * a1.scale);
                        const ny = (dx / len) * (2.8 * a1.scale);

                        ctx.strokeStyle = 'rgba(255, 255, 255, 0.45)';
                        ctx.lineWidth = Math.max(1, 1.8 * a1.scale);

                        ctx.beginPath();
                        ctx.moveTo(a1.proj.x + nx, a1.proj.y + ny);
                        ctx.lineTo(a2.proj.x + nx, a2.proj.y + ny);
                        ctx.stroke();

                        ctx.beginPath();
                        ctx.moveTo(a1.proj.x - nx, a1.proj.y - ny);
                        ctx.lineTo(a2.proj.x - nx, a2.proj.y - ny);
                        ctx.stroke();
                    } else {
                        ctx.strokeStyle = 'rgba(255, 255, 255, 0.35)';
                        ctx.lineWidth = Math.max(1, 2.0 * a1.scale);
                        ctx.beginPath();
                        ctx.moveTo(a1.proj.x, a1.proj.y);
                        ctx.lineTo(a2.proj.x, a2.proj.y);
                        ctx.stroke();
                    }
                }
                ctx.restore();

                for (const atom of projectedAtoms) {
                    const style = ATOM_COLORS[atom.type] || ATOM_COLORS.C;
                    const r = Math.max(1.2, style.radius * atom.scale * wobbleScale);

                    ctx.save();
                    ctx.globalAlpha = alpha;

                    const halo = ctx.createRadialGradient(
                        atom.proj.x, atom.proj.y, 0,
                        atom.proj.x, atom.proj.y, r * 2.5
                    );
                    halo.addColorStop(0, style.fill);
                    halo.addColorStop(0.4, style.glow);
                    halo.addColorStop(1, 'rgba(0, 0, 0, 0)');

                    ctx.fillStyle = halo;
                    ctx.beginPath();
                    ctx.arc(atom.proj.x, atom.proj.y, r * 2.5, 0, Math.PI * 2);
                    ctx.fill();

                    ctx.fillStyle = style.fill;
                    ctx.beginPath();
                    ctx.arc(atom.proj.x, atom.proj.y, r, 0, Math.PI * 2);
                    ctx.fill();

                    ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
                    ctx.beginPath();
                    ctx.arc(atom.proj.x - r * 0.3, atom.proj.y - r * 0.3, r * 0.35, 0, Math.PI * 2);
                    ctx.fill();

                    ctx.restore();
                }
            }
        }

        // -------------------------------------------------------------
        // 3. WOBBLING SCENT DROPLETS & BIO-VESICLES
        // -------------------------------------------------------------
        class ScentDroplet {
            constructor() {
                this.reset(true);
            }

            reset(initial = false) {
                this.x = Math.random() * width;
                this.y = initial ? Math.random() * height : height + 60;
                this.radius = 28 + Math.random() * 38;
                this.vy = -(0.25 + Math.random() * 0.45);
                this.vx = (Math.random() - 0.5) * 0.3;

                this.harmonics = [
                    { freq: 2, speed: 1.8 + Math.random() * 1.2, amp: 0.14 + Math.random() * 0.1, phase: Math.random() * Math.PI },
                    { freq: 3, speed: 2.2 + Math.random() * 1.5, amp: 0.08 + Math.random() * 0.07, phase: Math.random() * Math.PI },
                    { freq: 5, speed: 3.0 + Math.random() * 1.8, amp: 0.04 + Math.random() * 0.04, phase: Math.random() * Math.PI }
                ];

                const palettes = [
                    { start: 'rgba(0, 242, 254, 0.18)', mid: 'rgba(79, 172, 254, 0.08)', rim: 'rgba(0, 242, 254, 0.35)' },
                    { start: 'rgba(240, 147, 251, 0.16)', mid: 'rgba(155, 81, 224, 0.07)', rim: 'rgba(240, 147, 251, 0.32)' },
                    { start: 'rgba(0, 245, 160, 0.15)', mid: 'rgba(0, 217, 245, 0.06)', rim: 'rgba(0, 245, 160, 0.3)' },
                    { start: 'rgba(255, 117, 140, 0.15)', mid: 'rgba(255, 126, 179, 0.06)', rim: 'rgba(255, 117, 140, 0.3)' }
                ];
                this.colors = palettes[Math.floor(Math.random() * palettes.length)];
                this.depth = 50 + Math.random() * 200;
            }

            update(time) {
                this.x += this.vx;
                this.y += this.vy;

                if (this.y < -this.radius * 2) {
                    this.reset();
                }
                if (this.x < -this.radius * 2) this.x = width + this.radius;
                if (this.x > width + this.radius * 2) this.x = -this.radius;
            }

            draw(ctx, time, mouseOffset) {
                const posX = this.x + mouseOffset.x * (25 * (this.depth / 200));
                const posY = this.y + mouseOffset.y * (20 * (this.depth / 200));
                const points = 48;

                ctx.save();
                ctx.beginPath();

                for (let i = 0; i <= points; i++) {
                    const theta = (i / points) * Math.PI * 2;
                    let r = this.radius;

                    for (const h of this.harmonics) {
                        r += this.radius * h.amp * Math.sin(theta * h.freq + time * h.speed + h.phase);
                    }

                    const px = posX + Math.cos(theta) * r;
                    const py = posY + Math.sin(theta) * r;

                    if (i === 0) {
                        ctx.moveTo(px, py);
                    } else {
                        ctx.lineTo(px, py);
                    }
                }
                ctx.closePath();

                const grad = ctx.createRadialGradient(posX - this.radius * 0.3, posY - this.radius * 0.3, 0, posX, posY, this.radius * 1.2);
                grad.addColorStop(0, this.colors.start);
                grad.addColorStop(0.65, this.colors.mid);
                grad.addColorStop(1, 'rgba(0, 0, 0, 0)');

                ctx.fillStyle = grad;
                ctx.fill();

                ctx.strokeStyle = this.colors.rim;
                ctx.lineWidth = 1.2;
                ctx.stroke();

                ctx.restore();
            }
        }

        // -------------------------------------------------------------
        // 4. AMBIENT OLFACTORY SENSORY PARTICLES
        // -------------------------------------------------------------
        class SensoryParticle {
            constructor() {
                this.reset(true);
            }

            reset(initial = false) {
                this.x = Math.random() * width;
                this.y = initial ? Math.random() * height : height + 10;
                this.z = Math.random() * 250;
                this.vx = (Math.random() - 0.5) * 0.35;
                this.vy = -(0.2 + Math.random() * 0.4);
                this.baseRadius = 1.0 + Math.random() * 2.2;
                this.phase = Math.random() * Math.PI * 2;
                this.twinkleSpeed = 1.5 + Math.random() * 2.0;

                const hues = ['#00f2fe', '#4facfe', '#a8edea', '#f093fb', '#00f5a0'];
                this.color = hues[Math.floor(Math.random() * hues.length)];
            }

            update(time) {
                this.x += this.vx;
                this.y += this.vy;
                if (this.y < -15) this.reset();
                if (this.x < -15) this.x = width + 10;
                if (this.x > width + 15) this.x = -10;
            }

            draw(ctx, time, mouseOffset) {
                const posX = this.x + mouseOffset.x * 20;
                const posY = this.y + mouseOffset.y * 15;
                const alpha = 0.25 + 0.35 * Math.sin(time * this.twinkleSpeed + this.phase);

                ctx.save();
                ctx.globalAlpha = Math.max(0, Math.min(1, alpha));
                ctx.fillStyle = this.color;
                ctx.beginPath();
                ctx.arc(posX, posY, this.baseRadius, 0, Math.PI * 2);
                ctx.fill();
                ctx.restore();
            }
        }

        // -------------------------------------------------------------
        // INITIALIZATION & SCENE SETUP
        // -------------------------------------------------------------
        const dna = new DNAHelix();

        const molecules = [
            new MoleculeInstance(0, width * 0.18, height * 0.25, 40),
            new MoleculeInstance(1, width * 0.42, height * 0.78, 80),
            new MoleculeInstance(2, width * 0.15, height * 0.72, -30),
            new MoleculeInstance(3, width * 0.62, height * 0.22, 100),
            new MoleculeInstance(0, width * 0.78, height * 0.85, 20)
        ];

        const droplets = Array.from({ length: 7 }, () => new ScentDroplet());
        const particles = Array.from({ length: 45 }, () => new SensoryParticle());

        // -------------------------------------------------------------
        // MAIN ANIMATION LOOP
        // -------------------------------------------------------------
        let lastTime = 0;
        const mouseSmoothed = { x: 0, y: 0 };

        function render(timestamp) {
            const time = timestamp * 0.001;
            lastTime = timestamp;

            mouseSmoothed.x += (mouse.targetX - mouseSmoothed.x) * 0.06;
            mouseSmoothed.y += (mouse.targetY - mouseSmoothed.y) * 0.06;

            ctx.clearRect(0, 0, width, height);

            const bgGrad = ctx.createRadialGradient(
                width * 0.5 + mouseSmoothed.x * 50,
                height * 0.4 + mouseSmoothed.y * 40,
                50,
                width * 0.5,
                height * 0.5,
                Math.max(width, height) * 0.8
            );
            bgGrad.addColorStop(0, 'rgba(15, 23, 42, 0.85)');
            bgGrad.addColorStop(0.4, 'rgba(9, 13, 26, 0.95)');
            bgGrad.addColorStop(1, 'rgba(3, 7, 18, 1)');

            ctx.fillStyle = bgGrad;
            ctx.fillRect(0, 0, width, height);

            // 1. Liquid wobbling scent droplets
            for (const droplet of droplets) {
                droplet.update(time);
                droplet.draw(ctx, time, mouseSmoothed);
            }

            // 2. Ambient olfactory sensory dust
            for (const particle of particles) {
                particle.update(time);
                particle.draw(ctx, time, mouseSmoothed);
            }

            // 3. Rotating 3D DNA Double Helix
            dna.update(time);
            dna.draw(ctx, time, mouseSmoothed);

            // 4. 3D Floating & Wobbling Molecules
            for (const mol of molecules) {
                mol.update(time);
                mol.draw(ctx, time, mouseSmoothed);
            }

            requestAnimationFrame(render);
        }

        requestAnimationFrame(render);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
