// static/js/script.js

document.addEventListener('DOMContentLoaded', () => {
    console.log(">>> DOM cargado. Iniciando script.");

    // --- Elementos del DOM ---
    const lxInput = document.getElementById('lx');
    const lyInput = document.getElementById('ly');
    const tInput = document.getElementById('t');
    const kBaseInput = document.getElementById('k_base');
    const rthInput = document.getElementById('rth_heatsink');
    const tAmbientInput = document.getElementById('t_ambient_inlet');
    const qAirInput = document.getElementById('Q_total_m3_h');
    const placementArea = document.getElementById('placement-area');
    const placementColumn = document.querySelector('.placement-column');
    const addModuleButton = document.getElementById('add-module-button');
    const dynamicControlsContainer = document.getElementById('dynamic-controls');
    const noModulesMessage = document.getElementById('no-modules-message');
    const introSection = document.getElementById('intro-section');
    const scrollButton = document.getElementById('scroll-to-content-btn');
    const mainContent = document.getElementById('main-content');
    const mainForm = document.getElementById('simulation-main-form');
    const updateButton = document.getElementById('update-button');
    const statusDiv = document.getElementById('status');
    const loader = document.getElementById('loader');
    const resultsContainer = document.getElementById('results');
    const resultsSummaryDiv = document.getElementById('results-summary');
    // Interactivo (CANVAS)
    const interactivePlotContainer = document.getElementById('interactive-plot-container');
    const interactivePlotArea = document.getElementById('interactive-plot-area');
    const interactivePlotCanvas = document.getElementById('interactive-plot-canvas');
    const plotTooltip = document.getElementById('plot-tooltip');
    const interactivePlotPlaceholder = document.getElementById('interactive-plot-placeholder');
    const resetViewButton = document.getElementById('reset-view-button');
    // Estático Combinado Base/Aire
    const combinedPlotArea = document.getElementById('combined-plot-area');
    const combinedPlotImg = document.getElementById('combined-plot-img');
    const combinedPlotPlaceholder = document.getElementById('combined-plot-placeholder');
    // Estático Zoom
    const zoomPlotArea = document.getElementById('zoom-plot-area');
    const zoomPlotImg = document.getElementById('zoom-plot-img');
    const zoomPlotPlaceholder = document.getElementById('zoom-plot-placeholder');
    // PDF Modal
    const viewPdfButton = document.getElementById('view-pdf-button');
    const pdfModal = document.getElementById('pdf-modal');
    const pdfIframe = document.getElementById('pdf-iframe');
    const closePdfModalButton = document.getElementById('close-pdf-modal');
    const pdfDownloadLink = document.getElementById('pdf-download-link');

    // --- Contexto y Estado del Canvas Interactivo ---
    let ctx = interactivePlotCanvas ? interactivePlotCanvas.getContext('2d') : null;
    let baseImage = null; // La imagen original del plot térmico
    let currentScale = 1.0;
    let translateX = 0;
    let translateY = 0;
    let isPanning = false;
    let lastPanX = 0;
    let lastPanY = 0;
    const zoomFactor = 1.1;
    const minScale = 0.2; // Límite mínimo de zoom out
    const maxScale = 15.0; // Límite máximo de zoom in

    console.log(">>> Verificando elementos DOM:", {
        // ... (resto de verificaciones)
        interactivePlotCanvas: !!interactivePlotCanvas,
        ctx: !!ctx,
        resetViewButton: !!resetViewButton
    });

    // --- Estado de la Aplicación ---
    let modules = []; let nextModuleIndex = 1;
    const moduleFootprint = { w: 0.062, h: 0.122 }; const moveStep = 0.005;
    // Mantener simData para los datos numéricos
    let simData = { T: null, xCoords: null, yCoords: null, lx: null, ly: null, nx: null, ny: null, hasInteractiveData: false };

    // --- Funciones Auxiliares (Módulos/Placement sin cambios) ---
    function updatePlacementAreaAppearance() { if (!lxInput || !lyInput || !placementArea || !placementColumn) { console.warn("updatePlacementAreaAppearance: Elementos necesarios no disponibles."); return; } const lx = parseFloat(lxInput.value) || 0.25; const ly = parseFloat(lyInput.value) || 0.35; const aspectRatio = lx > 1e-6 && ly > 1e-6 ? lx / ly : 1; const containerWidth = placementColumn.offsetWidth - (2 * 1) ; let areaWidth = containerWidth; let areaHeight = containerWidth / aspectRatio; const maxHeight = 350; if (areaHeight > maxHeight || areaHeight <= 0 || !isFinite(areaHeight)) { areaHeight = maxHeight; areaWidth = maxHeight * aspectRatio; } areaWidth = Math.min(containerWidth, areaWidth); areaWidth = Math.max(50, areaWidth); areaHeight = Math.max(50, areaHeight); if (!isFinite(areaWidth) || !isFinite(areaHeight)) { areaWidth = Math.min(containerWidth, 300); areaHeight = 150; } placementArea.style.width = `${areaWidth}px`; placementArea.style.height = `${areaHeight}px`; modules.forEach(updateModuleVisual); }
    function updateModuleVisual(module) { if (!module.visualElement || !module.controlElement || !lxInput || !lyInput || !placementArea) return; const lx = parseFloat(lxInput.value); const ly = parseFloat(lyInput.value); const canvasWidth = placementArea.offsetWidth; const canvasHeight = placementArea.offsetHeight; if (!lx || lx <= 1e-6 || !ly || ly <= 1e-6 || !canvasWidth || canvasWidth <= 0 || !canvasHeight || canvasHeight <= 0) { module.visualElement.style.display = 'none'; return; } module.visualElement.style.display = 'flex'; const moduleVisualWidth = (moduleFootprint.w / lx) * canvasWidth; const moduleVisualHeight = (moduleFootprint.h / ly) * canvasHeight; const moduleLeft = ((module.x - moduleFootprint.w / 2) / lx) * canvasWidth; const moduleTop = canvasHeight * (1 - (module.y + moduleFootprint.h / 2) / ly); module.visualElement.style.width = `${Math.max(5, moduleVisualWidth)}px`; module.visualElement.style.height = `${Math.max(5, moduleVisualHeight)}px`; module.visualElement.style.left = `${Math.max(0, Math.min(canvasWidth - moduleVisualWidth, moduleLeft))}px`; module.visualElement.style.top = `${Math.max(0, Math.min(canvasHeight - moduleVisualHeight, moduleTop))}px`; const coordDisplay = module.controlElement.querySelector('.coord-display span'); if (coordDisplay) { coordDisplay.textContent = `X: ${module.x.toFixed(3)}m, Y: ${module.y.toFixed(3)}m`; } }
    function addModule() { console.log(">>> addModule() - INICIO"); if (!lxInput || !lyInput || !placementArea || !dynamicControlsContainer || !noModulesMessage || !statusDiv) { console.error("!!! addModule() - ERROR: Faltan elementos del DOM necesarios."); const msg = "Error interno: No se pueden añadir módulos (elementos DOM faltantes)."; if(statusDiv) showStatus(msg, true); else console.error(msg); return; } console.log(">>> addModule() - Elementos necesarios OK."); const lx = parseFloat(lxInput.value); const ly = parseFloat(lyInput.value); console.log(`>>> addModule() - Lx=${lx}, Ly=${ly}`); if (isNaN(lx) || lx <= 0 || isNaN(ly) || ly <= 0) { console.warn(">>> addModule() - Lx o Ly inválidos."); showStatus("Error: Define Lx y Ly positivos antes de añadir módulos.", true); return; } const newModuleId = `Mod_${nextModuleIndex}`; console.log(`>>> addModule() - Creando ${newModuleId}`); const initialX = Math.max(moduleFootprint.w / 2, Math.min(lx - moduleFootprint.w / 2, lx / 2)); const initialY = Math.max(moduleFootprint.h / 2, Math.min(ly - moduleFootprint.h / 2, ly / 2)); const newModule = { id: newModuleId, index: nextModuleIndex, x: initialX, y: initialY, powers: { 'IGBT1': 170.0, 'Diode1': 330.0, 'IGBT2': 170.0, 'Diode2': 330.0 }, visualElement: null, controlElement: null }; try { const visualDiv = document.createElement('div'); visualDiv.className = 'module-block'; visualDiv.dataset.moduleId = newModuleId; visualDiv.textContent = `M${newModule.index}`; visualDiv.title = `${newModuleId} (Center: X=${initialX.toFixed(3)}, Y=${initialY.toFixed(3)})`; placementArea.appendChild(visualDiv); newModule.visualElement = visualDiv; const controlFieldset = document.createElement('fieldset'); controlFieldset.className = 'module-controls'; controlFieldset.dataset.moduleId = newModuleId; controlFieldset.innerHTML = `<legend>${newModuleId}</legend><div class="coord-display">Center: <span>X: ${newModule.x.toFixed(3)}m, Y: ${newModule.y.toFixed(3)}m</span></div><div class="move-buttons"> Move: <button type="button" data-direction="left" title="Mover Izquierda (-X)">←</button> <button type="button" data-direction="right" title="Mover Derecha (+X)">→</button> <button type="button" data-direction="down" title="Mover Abajo (-Y)">↓</button> <button type="button" data-direction="up" title="Mover Arriba (+Y)">↑</button> </div><div>Power (W):</div>${Object.keys(newModule.powers).map(chipSuffix => `<div class="chip-input-group"><label for="${newModuleId}_${chipSuffix}">${chipSuffix}:</label><input type="number" id="${newModuleId}_${chipSuffix}" name="${newModuleId}_${chipSuffix}" value="${newModule.powers[chipSuffix].toFixed(1)}" step="1.0" min="0" required data-chip-suffix="${chipSuffix}"></div>`).join('')}<button type="button" data-action="delete" title="Eliminar Módulo ${newModuleId}">X Delete</button>`; dynamicControlsContainer.appendChild(controlFieldset); newModule.controlElement = controlFieldset; controlFieldset.addEventListener('click', handleModuleControlClick); controlFieldset.addEventListener('input', handlePowerInputChange); modules.push(newModule); nextModuleIndex++; updateModuleVisual(newModule); noModulesMessage.style.display = 'none'; showStatus(`Módulo ${newModuleId} añadido.`); console.log(`>>> addModule() - ${newModuleId} añadido exitosamente.`); } catch (error) { console.error("!!! addModule() - Error durante creación DOM:", error); showStatus("Error interno al crear el módulo.", true); } }
    function deleteModule(moduleId) { const moduleIndex = modules.findIndex(m => m.id === moduleId); if (moduleIndex > -1) { const { visualElement, controlElement } = modules[moduleIndex]; if (visualElement) visualElement.remove(); if (controlElement) controlElement.remove(); modules.splice(moduleIndex, 1); showStatus(`Módulo ${moduleId} eliminado.`); if (modules.length === 0 && noModulesMessage) { noModulesMessage.style.display = 'block'; } } else { console.warn(`Intento de eliminar módulo no encontrado: ${moduleId}`); } }
    function moveModule(moduleId, direction) { const module = modules.find(m => m.id === moduleId); if (!module || !lxInput || !lyInput) return; const lx = parseFloat(lxInput.value); const ly = parseFloat(lyInput.value); if (isNaN(lx) || lx <= 0 || isNaN(ly) || ly <= 0) { showStatus("Error: Lx y Ly deben ser positivos para mover.", true); return; } let { x: newX, y: newY } = module; switch (direction) { case 'left': newX -= moveStep; break; case 'right': newX += moveStep; break; case 'up': newY += moveStep; break; case 'down': newY -= moveStep; break; } const halfW = moduleFootprint.w / 2; const halfH = moduleFootprint.h / 2; module.x = Math.max(halfW, Math.min(lx - halfW, newX)); module.y = Math.max(halfH, Math.min(ly - halfH, newY)); updateModuleVisual(module); if (module.visualElement) { module.visualElement.title = `${module.id} (Centro: X=${module.x.toFixed(3)}, Y=${module.y.toFixed(3)})`; } }
    function openPdfModal() { const pdfUrl = '/view_pdf'; if (pdfIframe) { pdfIframe.src = pdfUrl; } if (pdfDownloadLink) { pdfDownloadLink.href = pdfUrl; } if (pdfModal) { pdfModal.style.display = 'block'; } }
    function closePdfModal() { if (pdfModal) { pdfModal.style.display = 'none'; } if (pdfIframe) { pdfIframe.src = 'about:blank'; } }

    // --- Funciones Canvas y Transformación ---

    /** Redibuja el canvas con la imagen base, aplicando transformaciones */
    function redrawCanvas() {
        if (!ctx || !baseImage || !interactivePlotCanvas) return;

        ctx.save();
        // Limpiar canvas
        ctx.clearRect(0, 0, interactivePlotCanvas.width, interactivePlotCanvas.height);
        // Aplicar transformaciones
        ctx.translate(translateX, translateY);
        ctx.scale(currentScale, currentScale);
        // Dibujar imagen base
        ctx.drawImage(baseImage, 0, 0);
        ctx.restore();
    }

    /** Obtiene las coordenadas del ratón relativas al canvas */
    function getMousePos(canvas, evt) {
        const rect = canvas.getBoundingClientRect();
        return {
            x: evt.clientX - rect.left,
            y: evt.clientY - rect.top
        };
    }

    /** Convierte coordenadas del canvas (pantalla) a coordenadas de la imagen original */
    function canvasToImageCoords(canvasX, canvasY) {
        if (!baseImage) return { x: 0, y: 0 };
        const imgX = (canvasX - translateX) / currentScale;
        const imgY = (canvasY - translateY) / currentScale;
        return { x: imgX, y: imgY };
    }

    /** Convierte coordenadas de la imagen original a coordenadas físicas (metros) */
    function imageToPhysicalCoords(imgX, imgY) {
        if (!baseImage || !simData.hasInteractiveData || simData.lx == null || simData.ly == null) {
            return { x: null, y: null };
        }
        // Asumiendo que la imagen RAW generada por matplotlib tiene el origen abajo a la izquierda
        // y las dimensiones mapean directamente lx y ly
        const physicalX = (imgX / baseImage.width) * simData.lx;
        const physicalY = (1 - (imgY / baseImage.height)) * simData.ly; // Invertir Y
        return { x: physicalX, y: physicalY };
    }

    /** Actualiza y muestra el tooltip */
    function updateTooltip(canvasX, canvasY) {
        if (!plotTooltip || !simData.hasInteractiveData || !baseImage) {
            if(plotTooltip) plotTooltip.style.display = 'none';
            return;
        }

        const { x: imgX, y: imgY } = canvasToImageCoords(canvasX, canvasY);

        // Verificar si el punto está dentro de la imagen original
        if (imgX < 0 || imgX >= baseImage.width || imgY < 0 || imgY >= baseImage.height) {
            plotTooltip.style.display = 'none';
            return;
        }

        const { x: physicalX, y: physicalY } = imageToPhysicalCoords(imgX, imgY);

        if (physicalX === null || physicalY === null) {
            plotTooltip.style.display = 'none';
            return;
        }

        // Encontrar el índice más cercano en la matriz de datos
        const idx_i = simData.nx > 1 ? Math.round((physicalX / simData.lx) * (simData.nx - 1)) : 0;
        const idx_j = simData.ny > 1 ? Math.round((physicalY / simData.ly) * (simData.ny - 1)) : 0;
        const i = Math.max(0, Math.min(simData.nx - 1, idx_i));
        const j = Math.max(0, Math.min(simData.ny - 1, idx_j));

        let temp = 'N/A';
        if (simData.T && i < simData.T.length && j < simData.T[i].length && simData.T[i][j] !== undefined && simData.T[i][j] !== null) {
             temp = formatTemp(simData.T[i][j], 1);
        } else {
             console.warn(`Tooltip: Índice/valor inválido en T[${i}][${j}]`);
             // Si T es null o los índices están fuera, temp seguirá siendo 'N/A'
        }


        plotTooltip.innerHTML = `X: ${physicalX.toFixed(3)} m<br>Y: ${physicalY.toFixed(3)} m<br>T: ${temp} °C`;

        // Posicionar el tooltip cerca del cursor
        const areaRect = interactivePlotArea.getBoundingClientRect();
        let tooltipX = canvasX + 15;
        let tooltipY = canvasY + 15;

        const estimatedTooltipWidth = plotTooltip.offsetWidth || 120;
        const estimatedTooltipHeight = plotTooltip.offsetHeight || 50;

        if (canvasX + estimatedTooltipWidth + 15 > interactivePlotArea.clientWidth) {
            tooltipX = canvasX - estimatedTooltipWidth - 5;
        }
        if (canvasY + estimatedTooltipHeight + 15 > interactivePlotArea.clientHeight) {
            tooltipY = canvasY - estimatedTooltipHeight - 5;
        }
        tooltipX = Math.max(5, tooltipX);
        tooltipY = Math.max(5, tooltipY);

        plotTooltip.style.left = `${tooltipX}px`;
        plotTooltip.style.top = `${tooltipY}px`;
        plotTooltip.style.display = 'block';
    }

    /** Restablece el zoom y el pan a la vista inicial */
    function resetView() {
        currentScale = 1.0;
        translateX = 0;
        translateY = 0;
        if (ctx && baseImage) {
             setCanvasSize(); // Reajusta tamaño CSS y redibuja con la vista reseteada
        } else if (ctx && interactivePlotCanvas) {
             // Si no hay imagen, al menos limpiar el canvas
              ctx.clearRect(0, 0, interactivePlotCanvas.width, interactivePlotCanvas.height);
        }
        if (plotTooltip) plotTooltip.style.display = 'none';
        console.log(">>> Vista restablecida.");
    }


    /** Ajusta el tamaño del canvas según su contenedor CSS y redibuja */
    function setCanvasSize() {
        if (!interactivePlotCanvas || !interactivePlotArea || !ctx) return;
        const displayWidth = interactivePlotArea.clientWidth;
        const displayHeight = interactivePlotArea.clientHeight;

        if (displayWidth <= 0 || displayHeight <= 0) {
            console.warn("setCanvasSize: Área de plot con tamaño cero, no se redimensiona canvas.");
            return;
        }

        if (interactivePlotCanvas.width !== displayWidth || interactivePlotCanvas.height !== displayHeight) {
            interactivePlotCanvas.width = displayWidth;
            interactivePlotCanvas.height = displayHeight;
            console.log(`Canvas resized to: ${displayWidth}x${displayHeight}`);
             // Redibujar SIEMPRE que se cambie el tamaño del canvas
            if (baseImage) {
                 redrawCanvas();
            } else {
                // Si no hay imagen, limpiar el canvas por si acaso
                 ctx.clearRect(0, 0, interactivePlotCanvas.width, interactivePlotCanvas.height);
            }
        } else if (baseImage) {
            // Si el tamaño no cambió pero hay imagen, redibujar por si acaso (ej. al resetear)
             redrawCanvas();
        }

    }


    // --- Manejadores de Eventos ---
    if (scrollButton && mainContent) { scrollButton.addEventListener('click', () => mainContent.scrollIntoView({ behavior: 'smooth', block: 'start' })); }
    if (addModuleButton) { addModuleButton.addEventListener('click', addModule); } else { console.error("!!! Error Crítico: Botón #add-module-button NO ENCONTRADO."); }
    function handleModuleControlClick(event) { const button = event.target.closest('button'); if (!button) return; const fieldset = button.closest('fieldset[data-module-id]'); if (!fieldset) return; const moduleId = fieldset.dataset.moduleId; const direction = button.dataset.direction; const action = button.dataset.action; if (direction) { moveModule(moduleId, direction); } else if (action === 'delete') { if (confirm(`¿Seguro que quieres eliminar ${moduleId}?`)) { deleteModule(moduleId); } } }
    function handlePowerInputChange(event) { const input = event.target; if (input.tagName !== 'INPUT' || input.type !== 'number' || !input.dataset.chipSuffix) return; const fieldset = input.closest('fieldset[data-module-id]'); if (!fieldset) return; const moduleId = fieldset.dataset.moduleId; const chipSuffix = input.dataset.chipSuffix; const powerValue = parseFloat(input.value); const module = modules.find(m => m.id === moduleId); if (module && module.powers.hasOwnProperty(chipSuffix)) { if (!isNaN(powerValue) && powerValue >= 0) { module.powers[chipSuffix] = powerValue; input.classList.remove('input-error'); } else if (input.value !== '') { input.value = module.powers[chipSuffix].toFixed(1); showStatus(`Potencia inválida para ${chipSuffix}. Debe ser >= 0.`, true); input.classList.add('input-error');} else { module.powers[chipSuffix] = 0.0; /* Asignar 0 si se deja vacío */ input.classList.remove('input-error'); } } }
    if (lxInput) lxInput.addEventListener('input', updatePlacementAreaAppearance);
    if (lyInput) lyInput.addEventListener('input', updatePlacementAreaAppearance);

    // Eventos del Canvas Interactivo
    if (interactivePlotCanvas && ctx) {
        // --- Zoom con Rueda del Ratón ---
        interactivePlotCanvas.addEventListener('wheel', (event) => {
            event.preventDefault();
            if (!baseImage) return; // No hacer zoom si no hay imagen

            const { x: mouseX, y: mouseY } = getMousePos(interactivePlotCanvas, event);

            const delta = event.deltaY > 0 ? 1 / zoomFactor : zoomFactor;
            const newScale = Math.max(minScale, Math.min(maxScale, currentScale * delta));

            // Si la escala no cambia (llegó al límite), no hacer nada más
            if (newScale === currentScale) return;

            const scaleChange = newScale / currentScale;

            translateX = mouseX - (mouseX - translateX) * scaleChange;
            translateY = mouseY - (mouseY - translateY) * scaleChange;

            currentScale = newScale;

            redrawCanvas();
            if (simData.hasInteractiveData) {
                updateTooltip(mouseX, mouseY);
            }
        });

        // --- Inicio del Pan (Arrastre) ---
        interactivePlotCanvas.addEventListener('mousedown', (event) => {
            if (event.button === 0 && baseImage) { // Botón izquierdo y hay imagen
                isPanning = true;
                lastPanX = event.clientX;
                lastPanY = event.clientY;
                interactivePlotArea.classList.add('panning');
                if (plotTooltip) plotTooltip.style.display = 'none';
            }
        });

        // --- Movimiento del Ratón (Pan y Tooltip) ---
        interactivePlotCanvas.addEventListener('mousemove', (event) => {
            const { x: mouseX, y: mouseY } = getMousePos(interactivePlotCanvas, event);

            if (isPanning) {
                const dx = event.clientX - lastPanX;
                const dy = event.clientY - lastPanY;
                translateX += dx;
                translateY += dy;
                lastPanX = event.clientX;
                lastPanY = event.clientY;
                redrawCanvas();
                if (plotTooltip) plotTooltip.style.display = 'none';
            } else if (baseImage && simData.hasInteractiveData) {
                updateTooltip(mouseX, mouseY);
            }
        });

        // --- Fin del Pan ---
        interactivePlotCanvas.addEventListener('mouseup', (event) => {
            if (event.button === 0) {
                isPanning = false;
                interactivePlotArea.classList.remove('panning');
                 // Podríamos mostrar tooltip al soltar si se quisiera
                 // if (baseImage && simData.hasInteractiveData) {
                 //     const { x: mouseX, y: mouseY } = getMousePos(interactivePlotCanvas, event);
                 //     updateTooltip(mouseX, mouseY);
                 // }
            }
        });

        // --- Ratón Sale del Canvas ---
        interactivePlotCanvas.addEventListener('mouseleave', () => {
            isPanning = false;
            interactivePlotArea.classList.remove('panning');
            if (plotTooltip) plotTooltip.style.display = 'none';
        });

        // --- Botón de Reset ---
        if (resetViewButton) {
             resetViewButton.addEventListener('click', resetView);
        }

    } else {
        console.warn("Canvas interactivo o contexto 2D no disponible.");
    }

    // Observador para redimensionar el canvas si su contenedor cambia
    let resizeObserver;
    if (typeof ResizeObserver !== 'undefined' && interactivePlotArea && interactivePlotCanvas) {
        resizeObserver = new ResizeObserver(entries => {
            // Usar requestAnimationFrame para evitar layout thrashing y asegurar que se ejecute
            // después de que el navegador haya estabilizado el layout
            window.requestAnimationFrame(() => {
                 // Solo ejecutar si el tamaño realmente cambió para evitar bucles innecesarios
                 const entry = entries[0];
                 const { width, height } = entry.contentRect;
                 if (interactivePlotCanvas.width !== Math.round(width) || interactivePlotCanvas.height !== Math.round(height)) {
                    console.log("ResizeObserver detectó cambio, llamando a setCanvasSize");
                    setCanvasSize();
                 }
            });
        });
        resizeObserver.observe(interactivePlotArea);
    } else {
        console.warn("ResizeObserver no soportado o elementos no encontrados.");
        // Fallback: ajustar tamaño en el evento resize de window
        let resizeTimeoutCanvas;
        window.addEventListener('resize', () => {
             clearTimeout(resizeTimeoutCanvas);
             resizeTimeoutCanvas = setTimeout(() => {
                 console.log("Window resize event, llamando a setCanvasSize");
                 setCanvasSize();
             }, 150); // Debounce
        });
    }


    // Resize general para placement area (sin cambios)
    let resizeTimeoutPlacement;
    window.addEventListener('resize', () => {
        clearTimeout(resizeTimeoutPlacement);
        resizeTimeoutPlacement = setTimeout(updatePlacementAreaAppearance, 150);
    });

    // PDF Modal (sin cambios)
    if (viewPdfButton && pdfModal && pdfIframe && closePdfModalButton) { viewPdfButton.addEventListener('click', openPdfModal); closePdfModalButton.addEventListener('click', closePdfModal); pdfModal.addEventListener('click', (event) => { if (event.target === pdfModal) closePdfModal(); }); document.addEventListener('keydown', (event) => { if (event.key === 'Escape' && pdfModal.style.display === 'block') closePdfModal(); }); } else { console.warn('Elementos del modal PDF no encontrados.'); }

    // --- Submit de la Simulación ---
    if (mainForm) {
        mainForm.addEventListener('submit', async (event) => {
            event.preventDefault();
            console.log(">>> Submit Simulación - INICIO");
            resetResultDisplays(); // Limpiar canvas, datos, etc.
            if (modules.length === 0) {
                showStatus('Error: No hay módulos añadidos.', true);
                if(resultsSummaryDiv) resultsSummaryDiv.innerHTML = '<p class="status-error-text">Añada al menos un módulo.</p>';
                showPlaceholders();
                return;
            }
            setLoadingState(true);
            if(resultsSummaryDiv) resultsSummaryDiv.innerHTML = '<p>Starting simulation...</p>';
            showPlaceholders(); // Mostrar placeholders mientras carga

            // ... (Validación de datos y creación de dataToSend igual que antes) ...
            const currentHeatsinkParams = { lx: lxInput?.value, ly: lyInput?.value, t: tInput?.value, k_base: kBaseInput?.value, rth_heatsink: rthInput?.value };
            const currentEnvironmentParams = { t_ambient_inlet: tAmbientInput?.value, Q_total_m3_h: qAirInput?.value };
            const currentModuleDefinitions = []; const currentPowers = {}; let errorMessages = [];
            // Validaciones
            if (isNaN(parseFloat(currentHeatsinkParams.lx)) || parseFloat(currentHeatsinkParams.lx) <= 0) errorMessages.push("Lx");
            if (isNaN(parseFloat(currentHeatsinkParams.ly)) || parseFloat(currentHeatsinkParams.ly) <= 0) errorMessages.push("Ly");
            if (isNaN(parseFloat(currentHeatsinkParams.t)) || parseFloat(currentHeatsinkParams.t) <= 0) errorMessages.push("t");
            if (isNaN(parseFloat(currentHeatsinkParams.k_base)) || parseFloat(currentHeatsinkParams.k_base) <= 1e-9) errorMessages.push("k");
            if (isNaN(parseFloat(currentHeatsinkParams.rth_heatsink)) || parseFloat(currentHeatsinkParams.rth_heatsink) <= 1e-9) errorMessages.push("Rth");
            if (isNaN(parseFloat(currentEnvironmentParams.t_ambient_inlet))) errorMessages.push("T_in");
            if (isNaN(parseFloat(currentEnvironmentParams.Q_total_m3_h)) || parseFloat(currentEnvironmentParams.Q_total_m3_h) <= 1e-9) errorMessages.push("Q");
            modules.forEach(module => {
                currentModuleDefinitions.push({ id: module.id, center_x: module.x, center_y: module.y });
                for (const chipSuffix in module.powers) {
                    const powerVal = module.powers[chipSuffix];
                    const inputElem = module.controlElement?.querySelector(`input[data-chip-suffix="${chipSuffix}"]`);
                    if (isNaN(powerVal) || powerVal < 0) {
                        errorMessages.push(`P(${module.id}_${chipSuffix})`);
                        if(inputElem) inputElem.classList.add('input-error');
                    } else {
                        if (inputElem) inputElem.classList.remove('input-error');
                    }
                    currentPowers[`${module.id}_${chipSuffix}`] = (powerVal >= 0 ? powerVal : 0).toString(); // Asegurar >= 0
                }
            });
            if (errorMessages.length > 0) {
                showStatus(`Error: Campos inválidos (${errorMessages.join(', ')})`, true);
                if(resultsSummaryDiv) resultsSummaryDiv.innerHTML = `<p class="status-error-text">Corrija los errores en los controles y reintente.</p>`;
                setLoadingState(false);
                showPlaceholders();
                return;
            }
            const dataToSend = { heatsink_params: currentHeatsinkParams, environment_params: currentEnvironmentParams, powers: currentPowers, module_definitions: currentModuleDefinitions };

            console.log(">>> Submit Simulación - Enviando:", JSON.stringify(dataToSend).substring(0, 500) + "..."); // Log truncado
            try {
                const response = await fetch('/update_simulation', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(dataToSend) });
                const results = await response.json();
                console.log(">>> Submit Simulación - Respuesta Recibida:", {status: response.status, ok: response.ok });
                if (!response.ok) { let errorMsg = `Error Servidor: ${response.status}`; if (results && results.message) errorMsg = results.message; throw new Error(errorMsg); }
                displayResults(results); // Muestra resultados en canvas y otros elementos
            } catch (error) {
                console.error('!!! Submit Simulación - Error en Fetch:', error);
                showStatus(`Error comunicación: ${error.message}`, true);
                if(resultsSummaryDiv) resultsSummaryDiv.innerHTML = `<p class="status-error-text">Error al obtener resultados: ${error.message}</p>`;
                showPlaceholders();
            } finally { setLoadingState(false); }
        });
    } else { console.error("!!! Error Crítico: Formulario #simulation-main-form NO ENCONTRADO."); }

    // --- Funciones de UI (ACTUALIZADAS) ---

    /** Oculta imágenes y muestra placeholders (Adaptado para Canvas) */
    function showPlaceholders() {
        if (interactivePlotCanvas) interactivePlotCanvas.style.display = 'none';
        if (interactivePlotPlaceholder) {
            interactivePlotPlaceholder.style.display = 'block';
            interactivePlotPlaceholder.textContent = "The interactive map will appear here."; // Mensaje por defecto
        }
        if (resetViewButton) resetViewButton.style.display = 'none';

        if (combinedPlotImg) combinedPlotImg.style.display = 'none';
        if (combinedPlotPlaceholder) {
             combinedPlotPlaceholder.style.display = 'block';
             combinedPlotPlaceholder.textContent = "El gráfico combinado Base/Aire aparecerá aquí.";
        }
        if (zoomPlotImg) zoomPlotImg.style.display = 'none';
        if (zoomPlotPlaceholder) {
             zoomPlotPlaceholder.style.display = 'block';
             zoomPlotPlaceholder.textContent = "El gráfico de detalle por módulo aparecerá aquí.";
        }
    }

    /** Resetea el área de resultados y los datos de simulación (Adaptado para Canvas) */
    function resetResultDisplays() {
        if(resultsSummaryDiv) resultsSummaryDiv.innerHTML = '<p>Esperando nueva simulación...</p>';
        showPlaceholders();

        // Limpiar estado del canvas y datos
        baseImage = null;
        resetView(); // Resetea scale/translate y limpia canvas si no hay imagen

        simData = { T: null, xCoords: null, yCoords: null, lx: null, ly: null, nx: null, ny: null, hasInteractiveData: false };

        // Limpiar plots estáticos
        if (combinedPlotImg) combinedPlotImg.src = '';
        if (zoomPlotImg) zoomPlotImg.src = '';
        if (plotTooltip) plotTooltip.style.display = 'none';
    }

    /** Muestra TODOS los resultados (Adaptado para Canvas) */
    function displayResults(results) {
         if (!resultsSummaryDiv || !interactivePlotCanvas || !interactivePlotPlaceholder || !interactivePlotArea ||
             !combinedPlotImg || !combinedPlotPlaceholder || !zoomPlotImg || !zoomPlotPlaceholder || !ctx || !resetViewButton) {
             console.error("!!! displayResults - Faltan elementos del DOM para mostrar resultados.");
             showStatus("Error interno al mostrar resultados.", true);
             return;
         }
         resetResultDisplays(); // Limpia antes de mostrar nuevos resultados
         let summaryHtml = `<h3>Summary</h3>`;

         if (results.status === 'Error' || !results.status || results.status.startsWith('Processing')) {
             const errorMsg = results.error_message || 'Error desconocido o simulación no completada.';
             summaryHtml += `<p class="status-error-text">Error en Simulación: ${errorMsg}</p>`;
             showStatus(`Error: ${errorMsg}`, true);
             showPlaceholders(); // Muestra placeholders de error
             if(interactivePlotPlaceholder) interactivePlotPlaceholder.textContent = "Error en simulación.";
             if(combinedPlotPlaceholder) combinedPlotPlaceholder.textContent = "Error en simulación.";
             if(zoomPlotPlaceholder) zoomPlotPlaceholder.textContent = "Error en simulación.";
         } else { // Éxito ('Success' o 'Success_NoInteractiveData')
             showStatus("Simulation completed.", false);
             // Construir resumen HTML (igual que antes)
             summaryHtml += `<p>Convergence: <span class="result-value ${results.convergence ? 'status-success-text' : 'status-error-text'}">${results.convergence ? 'Sí' : 'No'}</span> (Iter.: <span class="result-value">${results.iterations ?? 'N/A'}</span>)</p><p>T Máx Base: <span class="result-value">${formatTemp(results.t_max_base)}</span> °C</p><p>T Mean Base: <span class="result-value">${formatTemp(results.t_avg_base)}</span> °C</p><p>T Air Outlet: <span class="result-value">${formatTemp(results.t_air_outlet)}</span> °C</p><p>T Máx Juntion: <span class="result-value">${formatTemp(results.t_max_junction)}</span> °C (${results.t_max_junction_chip || 'N/A'})</p><p>T Máx NTC: <span class="result-value">${formatTemp(results.t_max_ntc)}</span> °C</p>${results.simulation_time_s !== undefined ? `<p>T. Sim.: <span class="result-value">${results.simulation_time_s.toFixed(2)}</span> s</p>` : ''}`;
             summaryHtml += `<h3>Results by module</h3><div id="module-results">`;
             if (results.module_results?.length > 0) { results.module_results.forEach(mod => { const modId = mod?.id || '?'; const tNtc = formatTemp(mod?.t_ntc); summaryHtml += `<div><b>${modId}:</b> T_NTC≈${tNtc}°C<br/>`; if (mod?.chips?.length > 0) { summaryHtml += '<ul>'; mod.chips.forEach(chip => { const chipSuffix = chip?.suffix || '?'; const tBase = formatTemp(chip?.t_base); const tJ = formatTemp(chip?.tj); summaryHtml += `<li>${chipSuffix}: T<sub>base</sub>≈${tBase}°C, T<sub>j</sub>=${tJ}°C</li>`; }); summaryHtml += '</ul>'; } else { summaryHtml += '<span class="details-na">(No chip data)</span>'; } summaryHtml += `</div>`; }); } else if (modules.length > 0) { summaryHtml += '<p>No se recibieron resultados detallados por módulo.</p>'; } else { summaryHtml += '<p>Simulación realizada sin módulos.</p>'; } summaryHtml += `</div>`;

             // --- Plot Interactivo (Cargar en Canvas) ---
             const hasInteractiveData = results.temperature_matrix && results.x_coordinates && results.y_coordinates &&
                                       results.sim_lx != null && results.sim_ly != null && results.sim_nx != null && results.sim_ny != null;

             // Reset simData antes de procesar nuevos resultados
             simData = { T: null, xCoords: null, yCoords: null, lx: null, ly: null, nx: null, ny: null, hasInteractiveData: false };

             if (results.plot_interactive_raw_uri) {
                 interactivePlotPlaceholder.style.display = 'block';
                 interactivePlotPlaceholder.textContent = 'Cargando mapa interactivo...';
                 interactivePlotCanvas.style.display = 'none';
                 resetViewButton.style.display = 'none';

                 baseImage = new Image();
                 baseImage.onload = () => {
                     console.log(">>> Imagen base para Canvas cargada.");
                     interactivePlotCanvas.style.display = 'block';
                     interactivePlotPlaceholder.style.display = 'none';
                     resetViewButton.style.display = 'inline-block';

                     setCanvasSize(); // Ajusta tamaño canvas y redibuja con vista reseteada
                     // resetView(); // resetView ahora llama a setCanvasSize, que redibuja

                     if (hasInteractiveData) {
                         console.log(">>> Datos interactivos RECIBIDOS.");
                         simData = { T: results.temperature_matrix, xCoords: results.x_coordinates, yCoords: results.y_coordinates, lx: results.sim_lx, ly: results.sim_ly, nx: results.sim_nx, ny: results.sim_ny, hasInteractiveData: true };
                         interactivePlotArea.style.cursor = 'crosshair'; // Cambiar cursor SÓLO si hay datos
                     } else {
                         console.warn(">>> Datos interactivos NO recibidos. Tooltip desactivado.");
                         interactivePlotArea.style.cursor = 'grab'; // Cursor por defecto si no hay datos
                         if (results.status === 'Success') { // Mostrar advertencia si faltan datos pero la simulación fue exitosa
                             showStatus("Simulación completada (datos interactivos no disponibles).", false); // No marcar como error
                         }
                     }
                 };
                 baseImage.onerror = () => {
                     console.error("!!! Error al cargar la imagen base para el Canvas.");
                     showPlaceholders();
                     interactivePlotPlaceholder.textContent = "Error al cargar mapa interactivo.";
                     if(resetViewButton) resetViewButton.style.display = 'none';
                     showStatus("Error cargando imagen del mapa.", true);
                 };
                 baseImage.src = results.plot_interactive_raw_uri;

             } else { // No hay URI para el plot interactivo
                 interactivePlotCanvas.style.display = 'none';
                 interactivePlotPlaceholder.style.display = 'block';
                 interactivePlotPlaceholder.textContent = "Mapa interactivo no disponible.";
                 if(resetViewButton) resetViewButton.style.display = 'none';
                 interactivePlotArea.style.cursor = 'default'; // Cursor normal
                 console.warn("Falta plot_interactive_raw_uri.");
             }

             // --- Plots Estáticos (Carga sin cambios) ---
             if (results.plot_base_data_uri) {
                 combinedPlotImg.src = results.plot_base_data_uri;
                 combinedPlotImg.style.display = 'block';
                 combinedPlotPlaceholder.style.display = 'none';
             } else {
                 combinedPlotImg.style.display = 'none';
                 combinedPlotPlaceholder.style.display = 'block';
                 combinedPlotPlaceholder.textContent = "Gráfico combinado Base/Aire no disponible.";
             }
             if (results.plot_zoom_data_uri) {
                 zoomPlotImg.src = results.plot_zoom_data_uri;
                 zoomPlotImg.style.display = 'block';
                 zoomPlotPlaceholder.style.display = 'none';
             } else {
                 zoomPlotImg.style.display = 'none';
                 zoomPlotPlaceholder.style.display = 'block';
                 zoomPlotPlaceholder.textContent = "Detalle por módulo no disponible.";
             }
         }
         // Actualizar el contenido del resumen al final
         resultsSummaryDiv.innerHTML = summaryHtml;
    }


    // --- Funciones de UI (sin cambios) ---
    /** Formatea temperatura */
    function formatTemp(value, precision = 1) { const num = parseFloat(value); if (value === null || value === undefined || isNaN(num)) { return 'N/A'; } return num.toFixed(precision); }
    /** Controla estado visual de carga */
    function setLoadingState(isLoading) { if (isLoading) { showStatus('Calculating...', false, true); if (loader) loader.style.display = 'inline-block'; if (updateButton) updateButton.disabled = true; document.body.classList.add('loading'); } else { if (loader) loader.style.display = 'none'; if (updateButton) updateButton.disabled = false; document.body.classList.remove('loading'); } }
     /** Muestra mensaje en barra de estado */
    function showStatus(message, isError = false, isLoading = false) { if (!statusDiv) return; statusDiv.textContent = message; statusDiv.className = 'status-neutral'; if (isLoading) { statusDiv.classList.add('status-loading'); } else if (isError) { statusDiv.classList.add('status-error'); } else { statusDiv.classList.add('status-success'); } }

    // --- Inicialización al Cargar la Página ---
    console.log(">>> Ejecutando inicialización...");
    if(placementArea) { updatePlacementAreaAppearance(); }
    if (noModulesMessage) { noModulesMessage.style.display = modules.length === 0 ? 'block' : 'none'; }
    showStatus("Ready to simulate", false);
    resetResultDisplays(); // Asegura estado inicial limpio del canvas y datos
    if(resultsSummaryDiv) { resultsSummaryDiv.innerHTML = '<p>Añada módulos, ajuste parámetros y pulse "Actualizar Simulación".</p>'; }

    // Ajustar tamaño inicial del canvas (importante que se haga después de que CSS cargue)
    // setTimeout(setCanvasSize, 0); // Llamar después del renderizado inicial
     setCanvasSize(); // Llamar directamente puede funcionar si el contenedor tiene tamaño definido

    console.log(">>> Inicialización completada.");

}); // Fin DOMContentLoaded