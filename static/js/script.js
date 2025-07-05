// static/js/script.js

document.addEventListener('DOMContentLoaded', () => {
    console.log(">>> DOM cargado. Iniciando script.");

    // --- Elementos del DOM (Generales y Placement) ---
    const lxInput = document.getElementById('lx');
    const lyInput = document.getElementById('ly');
    const tInput = document.getElementById('t'); // Base thickness
    const kBaseInput = document.getElementById('k_base');
    const rthManualInput = document.getElementById('rth_heatsink_manual');
    const useManualRthCheckbox = document.getElementById('use_manual_rth_checkbox');
    const tAmbientInput = document.getElementById('t_ambient_inlet');
    const qAirInput = document.getElementById('Q_total_m3_h');
    const placementArea = document.getElementById('placement-area');
    const placementColumn = document.querySelector('.placement-column');
    const addModuleButton = document.getElementById('add-module-button');
    const dynamicControlsContainer = document.getElementById('dynamic-controls');
    const noModulesMessage = document.getElementById('no-modules-message');
    const mainForm = document.getElementById('simulation-main-form');
    const updateButton = document.getElementById('update-button');
    const statusDiv = document.getElementById('status');
    const loader = document.getElementById('loader');

    // --- Elementos del DOM (Parámetros de Aletas) ---
    const hFinInput = document.getElementById('h_fin');
    const tFinInput = document.getElementById('t_fin');
    const numFinsInput = document.getElementById('num_fins');
    const wHollowInput = document.getElementById('w_hollow');
    const hHollowInput = document.getElementById('h_hollow');
    const numHollowPerFinInput = document.getElementById('num_hollow_per_fin');

    // --- Elementos del DOM (Visualización Sección Transversal) ---
    const crossSectionCanvas = document.getElementById('heatsink-cross-section-canvas');
    const crossSectionCtx = crossSectionCanvas ? crossSectionCanvas.getContext('2d') : null;
    const crossSectionMessage = document.getElementById('cross-section-message');

    // --- Elementos del DOM (Resultados y Plot Interactivo) ---
    const resultsSummaryDiv = document.getElementById('results-summary');
    const interactivePlotCanvas = document.getElementById('interactive-plot-canvas');
    const interactivePlotArea = document.getElementById('interactive-plot-area');
    const plotTooltip = document.getElementById('plot-tooltip');
    const interactivePlotPlaceholder = document.getElementById('interactive-plot-placeholder');
    const resetViewButton = document.getElementById('reset-view-button');
    const combinedPlotImg = document.getElementById('combined-plot-img');
    const combinedPlotPlaceholder = document.getElementById('combined-plot-placeholder');
    const zoomPlotImg = document.getElementById('zoom-plot-img');
    const zoomPlotPlaceholder = document.getElementById('zoom-plot-placeholder');

    // --- Elementos del DOM (Modal PDF) ---
    const viewPdfButton = document.getElementById('view-pdf-button');
    const pdfModal = document.getElementById('pdf-modal');
    const pdfIframe = document.getElementById('pdf-iframe');
    const closePdfModalButton = document.getElementById('close-pdf-modal');
    const pdfDownloadLink = document.getElementById('pdf-download-link');

    // +++ NUEVO ELEMENTO DEL DOM PARA EL BOTÓN DE REPORTE +++
    const generatePdfButton = document.getElementById('generate-pdf-report-button');


    // --- Contexto y Estado del Canvas Interactivo (Resultados) ---
    let ctxResults = interactivePlotCanvas ? interactivePlotCanvas.getContext('2d') : null;
    let baseImage = null;
    let currentScale = 1.0; let translateX = 0; let translateY = 0;
    let isPanning = false; let lastPanX = 0; let lastPanY = 0;
    const zoomFactor = 1.1; const minScale = 0.2; const maxScale = 15.0;

    // --- Estado de la Aplicación ---
    let modules = []; let nextModuleIndex = 1;
    const moduleFootprint = { w: 0.062, h: 0.122 }; const moveStep = 0.005;
    let simData = {
        temperature_matrix: null, x_coordinates: null, y_coordinates: null,
        sim_lx: null, sim_ly: null, sim_nx: null, sim_ny: null, hasInteractiveData: false
    };

    // +++ NUEVA VARIABLE PARA GUARDAR DATOS PARA EL PDF +++
    let lastSuccessfulSimData = null;


    // --- Funciones de Dibujo y Visualización ---

    function updatePlacementAreaAppearance() {
        if (!lxInput || !lyInput || !placementArea || !placementColumn) { console.warn("updatePlacementAreaAppearance: Elementos necesarios no disponibles."); return; }
        const lx = parseFloat(lxInput.value) || 0.25;
        const ly = parseFloat(lyInput.value) || 0.35;
        const aspectRatio = lx > 1e-6 && ly > 1e-6 ? lx / ly : 1;
        const containerWidth = placementColumn.offsetWidth - (2 * 1);
        let areaWidth = containerWidth;
        let areaHeight = containerWidth / aspectRatio;
        const maxHeight = 350;
        if (areaHeight > maxHeight || areaHeight <= 0 || !isFinite(areaHeight)) {
            areaHeight = maxHeight; areaWidth = maxHeight * aspectRatio;
        }
        areaWidth = Math.min(containerWidth, areaWidth);
        areaWidth = Math.max(50, areaWidth); areaHeight = Math.max(50, areaHeight);
        if (!isFinite(areaWidth) || !isFinite(areaHeight)) {
            areaWidth = Math.min(containerWidth, 300); areaHeight = 150;
        }
        placementArea.style.width = `${areaWidth}px`;
        placementArea.style.height = `${areaHeight}px`;

        drawFinsOnPlacementArea();
        modules.forEach(updateModuleVisual);
    }

    function drawFinsOnPlacementArea() { // Dibuja aletas en el área de placement (vista superior)
        if (!lxInput || !numFinsInput || !tFinInput || !placementArea) return;
        const existingFins = placementArea.querySelectorAll('.fin-visual');
        existingFins.forEach(fin => fin.remove());

        const lx = parseFloat(lxInput.value);
        const numFins = parseInt(numFinsInput.value);
        const tFin = parseFloat(tFinInput.value);

        if (isNaN(lx) || lx <= 0 || isNaN(numFins) || numFins <= 0 || isNaN(tFin) || tFin <= 0) return;

        const canvasWidthPx = placementArea.offsetWidth;
        const canvasHeightPx = placementArea.offsetHeight; // Representa Ly visualmente
        if (numFins * tFin >= lx) return; // Aletas ocupan todo el ancho

        const finVisualThicknessPx = (tFin / lx) * canvasWidthPx;
        const finVisualLengthPx = canvasHeightPx;

        let spacingTotalPx = canvasWidthPx - (numFins * finVisualThicknessPx);
        let spacingBetweenFinsPx = (numFins > 1) ? spacingTotalPx / (numFins -1) : 0; // Espacio entre centros de aletas si se distribuyen de borde a borde
                                                                            // O si se asume un espacio al inicio y al final:
        let initialSpacingPx = spacingTotalPx / (numFins + 1); // Si hay N+1 espacios (antes, entre, después)

        let currentXPx = initialSpacingPx;

        for (let i = 0; i < numFins; i++) {
            const finDiv = document.createElement('div');
            finDiv.className = 'fin-visual';
            finDiv.style.width = `${Math.max(1, finVisualThicknessPx)}px`;
            finDiv.style.height = `${finVisualLengthPx}px`;
            finDiv.style.left = `${currentXPx}px`;
            finDiv.style.top = `0px`;
            finDiv.title = `Fin (Top View)`;
            placementArea.appendChild(finDiv);
            currentXPx += finVisualThicknessPx + initialSpacingPx;
        }
    }

    function drawHeatsinkCrossSection() {
        if (!crossSectionCtx || !crossSectionCanvas) {
            console.warn("Canvas de sección transversal no disponible.");
            if (crossSectionMessage) crossSectionMessage.textContent = "Canvas no inicializado.";
            return;
        }

        const lx_m = parseFloat(lxInput.value);         // Ancho del disipador (eje X del canvas)
        const t_base_m = parseFloat(tInput.value);       // Espesor de la base (eje Y del canvas)
        const h_fin_m = parseFloat(hFinInput.value);     // Altura de aleta (eje Y del canvas)
        const t_fin_m = parseFloat(tFinInput.value);     // Espesor de aleta (eje X del canvas)
        const num_fins = parseInt(numFinsInput.value);   // Número de aletas

        const w_hollow_m = parseFloat(wHollowInput.value);
        const h_hollow_m = parseFloat(hHollowInput.value);
        const num_hollow_per_fin = parseInt(numHollowPerFinInput.value);

        crossSectionCtx.clearRect(0, 0, crossSectionCanvas.width, crossSectionCanvas.height);
        if (crossSectionMessage) crossSectionMessage.style.display = 'block';


        if (isNaN(lx_m) || lx_m <= 0 || isNaN(t_base_m) || t_base_m <= 0) {
            if(crossSectionMessage) crossSectionMessage.textContent = "Defina Lx y Espesor de Base válidos.";
            return;
        }

        const canvasWidth = crossSectionCanvas.width;
        const canvasHeight = crossSectionCanvas.height;
        const padding = 10; // Píxeles de padding alrededor del dibujo

        // Determinar la escala para que quepa el dibujo
        let totalWidth_m = lx_m;
        let totalHeight_m = t_base_m + ( (num_fins > 0 && !isNaN(h_fin_m) && h_fin_m > 0) ? h_fin_m : 0 );
        if (totalHeight_m <= 0) totalHeight_m = t_base_m; // Si no hay aletas, solo la base

        if (totalWidth_m <=0 || totalHeight_m <=0) {
             if(crossSectionMessage) crossSectionMessage.textContent = "Dimensiones totales inválidas para dibujar.";
            return;
        }

        const scaleX = (canvasWidth - 2 * padding) / totalWidth_m;
        const scaleY = (canvasHeight - 2 * padding) / totalHeight_m;
        const scale = Math.min(scaleX, scaleY) * 0.95; // Usar la escala más restrictiva y un pequeño margen

        if (scale <= 0 || !isFinite(scale)) {
             if(crossSectionMessage) crossSectionMessage.textContent = "Escala de dibujo inválida.";
            return;
        }
        if (crossSectionMessage) crossSectionMessage.style.display = 'none'; // Ocultar mensaje si se va a dibujar


        crossSectionCtx.save();
        // Centrar el dibujo
        const drawingWidthPx = totalWidth_m * scale;
        const drawingHeightPx = totalHeight_m * scale;
        const offsetX = (canvasWidth - drawingWidthPx) / 2;
        const offsetY = (canvasHeight - drawingHeightPx) / 2;
        crossSectionCtx.translate(offsetX, offsetY + drawingHeightPx); // Mover origen a abajo-izquierda del área de dibujo
        crossSectionCtx.scale(1, -1); // Invertir Y para que positivo sea hacia arriba

        // Colores
        const baseColor = "#A9A9A9"; // Gris oscuro
        const finColor = "#D3D3D3";  // Gris claro
        const hollowColor = "#FFFFFF"; // Blanco para el hueco
        const lineColor = "#666666";
        crossSectionCtx.lineWidth = 1;

        // Dibujar Base
        const baseThicknessPx = t_base_m * scale;
        const baseWidthPx = lx_m * scale;
        crossSectionCtx.fillStyle = baseColor;
        crossSectionCtx.fillRect(0, 0, baseWidthPx, baseThicknessPx);
        crossSectionCtx.strokeStyle = lineColor;
        crossSectionCtx.strokeRect(0, 0, baseWidthPx, baseThicknessPx);

        // Dibujar Aletas
        if (num_fins > 0 && !isNaN(h_fin_m) && h_fin_m > 0 && !isNaN(t_fin_m) && t_fin_m > 0) {
            if (num_fins * t_fin_m > lx_m + 1e-6) { // Pequeña tolerancia para errores de flotante
                if(crossSectionMessage) {
                    crossSectionMessage.textContent = "Advertencia: Las aletas ocupan más que el ancho Lx.";
                    crossSectionMessage.style.display = 'block';
                }
                 // No dibujar aletas si no caben, pero la base sí.
            } else {
                const finHeightPx = h_fin_m * scale;
                const finThicknessPx = t_fin_m * scale;

                // Distribución de aletas: asumimos que se distribuyen uniformemente
                // con el mismo espaciado en los bordes y entre ellas. N+1 espacios.
                const totalSpaceForFinsAndGaps = lx_m;
                const totalFinMaterialWidth = num_fins * t_fin_m;
                const totalGapWidth = totalSpaceForFinsAndGaps - totalFinMaterialWidth;

                let gapWidthPx = 0;
                if (num_fins > 0) { // Evitar división por cero si num_fins es 0 (aunque ya está en el if)
                    gapWidthPx = (totalGapWidth / (num_fins + 1)) * scale;
                }


                let currentX_px = gapWidthPx; // Empezar después del primer hueco

                for (let i = 0; i < num_fins; i++) {
                    crossSectionCtx.fillStyle = finColor;
                    crossSectionCtx.fillRect(currentX_px, baseThicknessPx, finThicknessPx, finHeightPx);
                    crossSectionCtx.strokeStyle = lineColor;
                    crossSectionCtx.strokeRect(currentX_px, baseThicknessPx, finThicknessPx, finHeightPx);

                    // Dibujar Huecos dentro de la aleta
                    if (num_hollow_per_fin > 0 && !isNaN(w_hollow_m) && w_hollow_m > 0 &&
                        !isNaN(h_hollow_m) && h_hollow_m > 0 &&
                        w_hollow_m < t_fin_m && h_hollow_m < h_fin_m) { // Hueco debe caber

                        const hollowWidthPx = w_hollow_m * scale;
                        const hollowHeightPx = h_hollow_m * scale;

                        // Distribuir huecos centrados en la aleta
                        const totalHollowMaterialHeight = num_hollow_per_fin * h_hollow_m;
                        const totalVerticalGapInFin = h_fin_m - totalHollowMaterialHeight;
                        let verticalGapHollowPx = 0;
                        if (num_hollow_per_fin > 0) {
                             verticalGapHollowPx = (totalVerticalGapInFin / (num_hollow_per_fin + 1)) * scale;
                        }

                        let currentY_hollow_px = baseThicknessPx + verticalGapHollowPx; // Y desde la base de la aleta

                        const horizontalOffsetHollowPx = (finThicknessPx - hollowWidthPx) / 2; // Centrar horizontalmente el hueco

                        for (let j = 0; j < num_hollow_per_fin; j++) {
                            crossSectionCtx.fillStyle = hollowColor;
                            crossSectionCtx.fillRect(currentX_px + horizontalOffsetHollowPx, currentY_hollow_px, hollowWidthPx, hollowHeightPx);
                            crossSectionCtx.strokeStyle = lineColor;
                            crossSectionCtx.strokeRect(currentX_px + horizontalOffsetHollowPx, currentY_hollow_px, hollowWidthPx, hollowHeightPx);
                            currentY_hollow_px += hollowHeightPx + verticalGapHollowPx;
                        }
                    }
                    currentX_px += finThicknessPx + gapWidthPx;
                }
            }
        } else {
            if(crossSectionMessage && num_fins > 0) crossSectionMessage.textContent = "Defina altura y espesor de aleta válidos.";
        }
        crossSectionCtx.restore();
    }


    function updateModuleVisual(module) {
        // ... (sin cambios, ya implementada)
        if (!module.visualElement || !module.controlElement || !lxInput || !lyInput || !placementArea) return;
        const lx = parseFloat(lxInput.value); const ly = parseFloat(lyInput.value);
        const canvasWidth = placementArea.offsetWidth; const canvasHeight = placementArea.offsetHeight;
        if (!lx || lx <= 1e-6 || !ly || ly <= 1e-6 || !canvasWidth || canvasWidth <= 0 || !canvasHeight || canvasHeight <= 0) { module.visualElement.style.display = 'none'; return; }
        module.visualElement.style.display = 'flex';
        const moduleVisualWidth = (moduleFootprint.w / lx) * canvasWidth;
        const moduleVisualHeight = (moduleFootprint.h / ly) * canvasHeight;
        const moduleLeftPx = ((module.x - moduleFootprint.w / 2) / lx) * canvasWidth;
        const moduleTopPx = canvasHeight * (1 - (module.y + moduleFootprint.h / 2) / ly);
        module.visualElement.style.width = `${Math.max(5, moduleVisualWidth)}px`;
        module.visualElement.style.height = `${Math.max(5, moduleVisualHeight)}px`;
        module.visualElement.style.left = `${Math.max(0, Math.min(canvasWidth - moduleVisualWidth, moduleLeftPx))}px`;
        module.visualElement.style.top = `${Math.max(0, Math.min(canvasHeight - moduleVisualHeight, moduleTopPx))}px`;

    if (module.xInput && document.activeElement !== module.xInput) {
        module.xInput.value = module.x.toFixed(3);
    }
    if (module.yInput && document.activeElement !== module.yInput) {
        module.yInput.value = module.y.toFixed(3);
    }
    }

    function addModule() { /* ... (sin cambios, ya implementada) ... */
        if (!lxInput || !lyInput || !placementArea || !dynamicControlsContainer || !noModulesMessage || !statusDiv) { console.error("addModule: Faltan elementos DOM."); showStatus("Error interno: No se pueden añadir módulos.", true); return; }
        const lx = parseFloat(lxInput.value); const ly = parseFloat(lyInput.value);
        if (isNaN(lx) || lx <= 0 || isNaN(ly) || ly <= 0) { showStatus("Error: Define Lx y Ly positivos antes de añadir módulos.", true); return; }
        const newModuleId = `Mod_${nextModuleIndex}`;
        const initialX = Math.max(moduleFootprint.w / 2, Math.min(lx - moduleFootprint.w / 2, lx / 2));
        const initialY = Math.max(moduleFootprint.h / 2, Math.min(ly - moduleFootprint.h / 2, ly / 2));
        const newModule = { id: newModuleId, index: nextModuleIndex, x: initialX, y: initialY, powers: { 'IGBT1': 170.0, 'Diode1': 330.0, 'IGBT2': 170.0, 'Diode2': 330.0 }, visualElement: null, controlElement: null };
        try {
            const visualDiv = document.createElement('div'); visualDiv.className = 'module-block'; visualDiv.dataset.moduleId = newModuleId; visualDiv.textContent = `M${newModule.index}`; visualDiv.title = `${newModuleId} (Center: X=${initialX.toFixed(3)}, Y=${initialY.toFixed(3)})`; placementArea.appendChild(visualDiv); newModule.visualElement = visualDiv;
            const controlFieldset = document.createElement('fieldset'); controlFieldset.className = 'module-controls'; controlFieldset.dataset.moduleId = newModuleId;
            controlFieldset.innerHTML = `<legend>${newModuleId}</legend><div class="move-buttons"> Move: <button type="button" data-direction="left" title="Mover Izquierda (-X)">←</button> <button type="button" data-direction="right" title="Mover Derecha (+X)">→</button> <button type="button" data-direction="down" title="Mover Abajo (-Y)">↓</button> <button type="button" data-direction="up" title="Mover Arriba (+Y)">↑</button> </div><div class="coord-inputs"><label for="${newModuleId}_x_input">X:</label><input type="number" id="${newModuleId}_x_input" step="0.001" style="width: 70px;" title="Set X coordinate"><label for="${newModuleId}_y_input">Y:</label><input type="number" id="${newModuleId}_y_input" step="0.001" style="width: 70px;" title="Set Y coordinate"></div><div>Power (W):</div>${Object.keys(newModule.powers).map(chipSuffix => `<div class="chip-input-group"><label for="${newModuleId}_${chipSuffix}">${chipSuffix}:</label><input type="number" id="${newModuleId}_${chipSuffix}" name="${newModuleId}_${chipSuffix}" value="${newModule.powers[chipSuffix].toFixed(1)}" step="1.0" min="0" required data-chip-suffix="${chipSuffix}"></div>`).join('')}<button type="button" data-action="delete" title="Eliminar Módulo ${newModuleId}">X Delete</button>`;
            dynamicControlsContainer.appendChild(controlFieldset); newModule.controlElement = controlFieldset;

            newModule.xInput = controlFieldset.querySelector(`#${newModuleId}_x_input`);
            newModule.yInput = controlFieldset.querySelector(`#${newModuleId}_y_input`);

            const currentLx = parseFloat(lxInput.value);
            const currentLy = parseFloat(lyInput.value);
            const halfW = moduleFootprint.w / 2;
            const halfH = moduleFootprint.h / 2;

            if (newModule.xInput) {
                newModule.xInput.min = halfW.toFixed(3);
                newModule.xInput.max = (currentLx - halfW).toFixed(3);
                newModule.xInput.value = newModule.x.toFixed(3);
            }
            if (newModule.yInput) {
                newModule.yInput.min = halfH.toFixed(3);
                newModule.yInput.max = (currentLy - halfH).toFixed(3);
                newModule.yInput.value = newModule.y.toFixed(3);
            }

            if (newModule.xInput) {
                newModule.xInput.addEventListener('change', () => handleCoordinateInputChange(newModule));
            }
            if (newModule.yInput) {
                newModule.yInput.addEventListener('change', () => handleCoordinateInputChange(newModule));
            }

            controlFieldset.addEventListener('click', handleModuleControlClick); controlFieldset.addEventListener('input', handlePowerInputChange);
            modules.push(newModule); nextModuleIndex++; updateModuleVisual(newModule); if (noModulesMessage) noModulesMessage.style.display = 'none'; showStatus(`Módulo ${newModuleId} añadido.`);
        } catch (error) { console.error("addModule: Error creando DOM:", error); showStatus("Error interno al crear módulo.", true); }
    }
    function deleteModule(moduleId) { /* ... (sin cambios, ya implementada) ... */
        const moduleIdx = modules.findIndex(m => m.id === moduleId);
        if (moduleIdx > -1) { const { visualElement, controlElement } = modules[moduleIdx]; if (visualElement) visualElement.remove(); if (controlElement) controlElement.remove(); modules.splice(moduleIdx, 1); showStatus(`Módulo ${moduleId} eliminado.`); if (modules.length === 0 && noModulesMessage) { noModulesMessage.style.display = 'block'; } }
    }
    function moveModule(moduleId, direction) { /* ... (sin cambios, ya implementada) ... */
        const module = modules.find(m => m.id === moduleId); if (!module || !lxInput || !lyInput) return;
        const lx = parseFloat(lxInput.value); const ly = parseFloat(lyInput.value);
        if (isNaN(lx) || lx <= 0 || isNaN(ly) || ly <= 0) { showStatus("Error: Lx y Ly deben ser positivos para mover.", true); return; }
        let { x: newX, y: newY } = module;
        switch (direction) { case 'left': newX -= moveStep; break; case 'right': newX += moveStep; break; case 'up': newY += moveStep; break; case 'down': newY -= moveStep; break; }
        const halfW = moduleFootprint.w / 2; const halfH = moduleFootprint.h / 2;
        module.x = Math.max(halfW, Math.min(lx - halfW, newX)); module.y = Math.max(halfH, Math.min(ly - halfH, newY));
        updateModuleVisual(module); if (module.visualElement) { module.visualElement.title = `${module.id} (Center: X=${module.x.toFixed(3)}, Y=${module.y.toFixed(3)})`; }
    }

    function handleCoordinateInputChange(module) {
        if (!module || !module.xInput || !module.yInput || !lxInput || !lyInput) {
            console.warn("handleCoordinateInputChange: Missing module or critical elements.");
            return;
        }

        const lx = parseFloat(lxInput.value);
        const ly = parseFloat(lyInput.value);

        if (isNaN(lx) || lx <= 0 || isNaN(ly) || ly <= 0) {
            showStatus("Error: Heatsink dimensions Lx and Ly must be positive.", true);
            // Revert to current module values if heatsink dimensions are invalid
            module.xInput.value = module.x.toFixed(3);
            module.yInput.value = module.y.toFixed(3);
            return;
        }

        let newX = parseFloat(module.xInput.value);
        let newY = parseFloat(module.yInput.value);

        let changed = false;

        if (isNaN(newX)) {
            newX = module.x; // Revert to old value if not a number
            module.xInput.value = module.x.toFixed(3);
        }
        if (isNaN(newY)) {
            newY = module.y; // Revert to old value if not a number
            module.yInput.value = module.y.toFixed(3);
        }

        const halfW = moduleFootprint.w / 2;
        const halfH = moduleFootprint.h / 2;

        // Validate and clamp X value
        const minX = halfW;
        const maxX = lx - halfW;
        if (newX < minX) {
            newX = minX;
            module.xInput.value = newX.toFixed(3);
        } else if (newX > maxX) {
            newX = maxX;
            module.xInput.value = newX.toFixed(3);
        }

        // Validate and clamp Y value
        const minY = halfH;
        const maxY = ly - halfH;
        if (newY < minY) {
            newY = minY;
            module.yInput.value = newY.toFixed(3);
        } else if (newY > maxY) {
            newY = maxY;
            module.yInput.value = newY.toFixed(3);
        }

        if (module.x !== newX) {
            module.x = newX;
            changed = true;
        }
        if (module.y !== newY) {
            module.y = newY;
            changed = true;
        }

        if (changed) {
            updateModuleVisual(module);
            if (module.visualElement) {
                module.visualElement.title = `${module.id} (Center: X=${module.x.toFixed(3)}, Y=${module.y.toFixed(3)})`;
            }
            showStatus(`Module ${module.id} position updated via input.`);
        }
    }

    function updateAllModuleCoordinateInputLimits() {
        if (!lxInput || !lyInput) return;
        const currentLx = parseFloat(lxInput.value);
        const currentLy = parseFloat(lyInput.value);

        if (isNaN(currentLx) || currentLx <= 0 || isNaN(currentLy) || currentLy <= 0) {
            // If heatsink dimensions are invalid, perhaps clear max or handle error
            // For now, we'll just avoid updating if dimensions are bad.
            return;
        }

        const halfW = moduleFootprint.w / 2;
        const halfH = moduleFootprint.h / 2;
        const maxX = currentLx - halfW;
        const maxY = currentLy - halfH;

        modules.forEach(module => {
            if (module.xInput) {
                module.xInput.max = maxX.toFixed(3);
                // Re-validate/clamp current X value if it exceeds new max
                if (module.x > maxX) {
                    module.x = maxX; // Update module property directly
                                     // handleCoordinateInputChange will be called by its own event if user types.
                                     // or we can call it programmatically if needed,
                                     // but updateModuleVisual will fix the display.
                }
            }
            if (module.yInput) {
                module.yInput.max = maxY.toFixed(3);
                // Re-validate/clamp current Y value if it exceeds new max
                if (module.y > maxY) {
                    module.y = maxY; // Update module property directly
                }
            }
            // After updating module.x and module.y, ensure visuals and inputs are correct
            // We need to ensure that if a module's x or y was changed here because it was out of bounds,
            // its input field AND visual display are updated.
            // Calling updateModuleVisual will update the input field value (due to changes in Step 2)
            // and the visual position.
            updateModuleVisual(module);
        });
    }

    function openPdfModal() { /* ... (sin cambios, ya implementada) ... */ const pdfUrl = '/view_pdf'; if (pdfIframe) pdfIframe.src = pdfUrl; if (pdfDownloadLink) pdfDownloadLink.href = pdfUrl; if (pdfModal) pdfModal.style.display = 'block'; }
    function closePdfModal() { /* ... (sin cambios, ya implementada) ... */ if (pdfModal) pdfModal.style.display = 'none'; if (pdfIframe) pdfIframe.src = 'about:blank'; }

    // --- Funciones Canvas Resultados y Transformación ---
    function redrawCanvasResults() { /* Renombrado para claridad */ if (!ctxResults || !interactivePlotCanvas) return; ctxResults.save(); ctxResults.clearRect(0, 0, interactivePlotCanvas.width, interactivePlotCanvas.height); if (baseImage) { ctxResults.translate(translateX, translateY); ctxResults.scale(currentScale, currentScale); ctxResults.drawImage(baseImage, 0, 0); } ctxResults.restore(); }
    function getMousePosResults(canvas, evt) { /* Renombrado */ const rect = canvas.getBoundingClientRect(); return { x: evt.clientX - rect.left, y: evt.clientY - rect.top }; }
    function canvasToImageCoords(canvasX, canvasY) { if (!baseImage) return { x: 0, y: 0 }; return { x: (canvasX - translateX) / currentScale, y: (canvasY - translateY) / currentScale }; }
    function imageToPhysicalCoords(imgX, imgY) { if (!baseImage || !simData.hasInteractiveData || simData.sim_lx == null || simData.sim_ly == null) { return { x: null, y: null }; } const physicalX = (imgX / baseImage.width) * simData.sim_lx; const physicalY = (1 - (imgY / baseImage.height)) * simData.sim_ly; return { x: physicalX, y: physicalY }; }
    function updateTooltip(canvasX, canvasY) { if (!plotTooltip || !simData.hasInteractiveData || !baseImage || !simData.temperature_matrix || !simData.x_coordinates || !simData.y_coordinates || simData.sim_lx == null || simData.sim_ly == null || simData.sim_nx == null || simData.sim_ny == null) { if(plotTooltip) plotTooltip.style.display = 'none'; return; } const { x: imgX, y: imgY } = canvasToImageCoords(canvasX, canvasY); if (imgX < 0 || imgX >= baseImage.width || imgY < 0 || imgY >= baseImage.height) { plotTooltip.style.display = 'none'; return; } const { x: physicalX, y: physicalY } = imageToPhysicalCoords(imgX, imgY); if (physicalX === null || physicalY === null) { plotTooltip.style.display = 'none'; return; } let idx_i = Math.round((physicalX / simData.sim_lx) * (simData.sim_nx - 1)); let idx_j = Math.round((physicalY / simData.sim_ly) * (simData.sim_ny - 1)); idx_i = Math.max(0, Math.min(simData.sim_nx - 1, idx_i)); idx_j = Math.max(0, Math.min(simData.sim_ny - 1, idx_j)); let tempValue = 'N/A'; if (simData.temperature_matrix && idx_i >= 0 && idx_i < simData.temperature_matrix.length && idx_j >= 0 && idx_j < simData.temperature_matrix[idx_i].length) { tempValue = simData.temperature_matrix[idx_i][idx_j]; } else { console.warn(`Tooltip: Índice fuera de rango T[${idx_i}][${idx_j}]`); } const tempStr = formatTemp(tempValue, 1); plotTooltip.innerHTML = `X: ${physicalX.toFixed(3)}m<br>Y: ${physicalY.toFixed(3)}m<br>T: ${tempStr}°C`; let tooltipX = canvasX + 15; let tooltipY = canvasY + 15; const estimatedTooltipWidth = plotTooltip.offsetWidth || 120; const estimatedTooltipHeight = plotTooltip.offsetHeight || 50; if (canvasX + estimatedTooltipWidth + 15 > interactivePlotArea.clientWidth) tooltipX = canvasX - estimatedTooltipWidth - 5; if (canvasY + estimatedTooltipHeight + 15 > interactivePlotArea.clientHeight) tooltipY = canvasY - estimatedTooltipHeight - 5; tooltipX = Math.max(5, Math.min(tooltipX, interactivePlotArea.clientWidth - estimatedTooltipWidth - 5)); tooltipY = Math.max(5, Math.min(tooltipY, interactivePlotArea.clientHeight - estimatedTooltipHeight - 5)); plotTooltip.style.left = `${tooltipX}px`; plotTooltip.style.top = `${tooltipY}px`; plotTooltip.style.display = 'block'; }
    function resetViewResults() { /* Renombrado */ currentScale = 1.0; translateX = 0; translateY = 0; if (ctxResults && interactivePlotCanvas) setCanvasSizeResults(); if (plotTooltip) plotTooltip.style.display = 'none'; }
    function setCanvasSizeResults() { /* Renombrado */ if (!interactivePlotCanvas || !interactivePlotArea || !ctxResults) return; const displayWidth = interactivePlotArea.clientWidth; const displayHeight = interactivePlotArea.clientHeight; if (displayWidth <= 0 || displayHeight <= 0) { console.warn("setCanvasSizeResults: Área de plot con tamaño cero."); return; } if (interactivePlotCanvas.width !== displayWidth || interactivePlotCanvas.height !== displayHeight) { interactivePlotCanvas.width = displayWidth; interactivePlotCanvas.height = displayHeight; } redrawCanvasResults(); }

    // --- Manejadores de Eventos ---
    document.getElementById('scroll-to-content-btn')?.addEventListener('click', () => document.getElementById('main-content').scrollIntoView({ behavior: 'smooth', block: 'start' }));
    if (addModuleButton) { addModuleButton.addEventListener('click', addModule); }
    function handleModuleControlClick(event) { const button = event.target.closest('button'); if (!button) return; const fieldset = button.closest('fieldset[data-module-id]'); if (!fieldset) return; const moduleId = fieldset.dataset.moduleId; const direction = button.dataset.direction; const action = button.dataset.action; if (direction) { moveModule(moduleId, direction); } else if (action === 'delete') { if (confirm(`¿Seguro que quieres eliminar ${moduleId}?`)) { deleteModule(moduleId); } } }
    function handlePowerInputChange(event) { const input = event.target; if (input.tagName !== 'INPUT' || input.type !== 'number' || !input.dataset.chipSuffix) return; const fieldset = input.closest('fieldset[data-module-id]'); if (!fieldset) return; const moduleId = fieldset.dataset.moduleId; const chipSuffix = input.dataset.chipSuffix; const powerValue = parseFloat(input.value); const module = modules.find(m => m.id === moduleId); if (module && module.powers.hasOwnProperty(chipSuffix)) { if (!isNaN(powerValue) && powerValue >= 0) { module.powers[chipSuffix] = powerValue; input.classList.remove('input-error'); } else if (input.value !== '') { input.value = module.powers[chipSuffix].toFixed(1); showStatus(`Potencia inválida para ${chipSuffix}. Debe ser >= 0.`, true); input.classList.add('input-error');} else { module.powers[chipSuffix] = 0.0; input.classList.remove('input-error'); } } }

    // Eventos para actualizar visualización de aletas en placement-area Y sección transversal
    const geometryInputs = [lxInput, lyInput, tInput, hFinInput, tFinInput, numFinsInput, wHollowInput, hHollowInput, numHollowPerFinInput];
    geometryInputs.forEach(input => {
        if (input) {
            input.addEventListener('input', () => {
                if (input === lxInput || input === lyInput) {
                    updateAllModuleCoordinateInputLimits(); // Call the new function
                }
                updatePlacementAreaAppearance(); // Existing call
                drawHeatsinkCrossSection();    // Existing call
            });
        }
    });

    // Eventos del Canvas de Resultados
    if (interactivePlotCanvas && ctxResults) {
        interactivePlotCanvas.addEventListener('wheel', (event) => { event.preventDefault(); if (!baseImage) return; const { x: mouseX, y: mouseY } = getMousePosResults(interactivePlotCanvas, event); const delta = event.deltaY > 0 ? 1 / zoomFactor : zoomFactor; const newScale = Math.max(minScale, Math.min(maxScale, currentScale * delta)); if (newScale === currentScale) return; const scaleChange = newScale / currentScale; translateX = mouseX - (mouseX - translateX) * scaleChange; translateY = mouseY - (mouseY - translateY) * scaleChange; currentScale = newScale; redrawCanvasResults(); if (simData.hasInteractiveData) { updateTooltip(mouseX, mouseY); } });
        interactivePlotCanvas.addEventListener('mousedown', (event) => { if (event.button === 0 && baseImage) { isPanning = true; lastPanX = event.clientX; lastPanY = event.clientY; interactivePlotArea.classList.add('panning'); if (plotTooltip) plotTooltip.style.display = 'none'; } });
        interactivePlotCanvas.addEventListener('mousemove', (event) => { const { x: mouseX, y: mouseY } = getMousePosResults(interactivePlotCanvas, event); if (isPanning) { const dx = event.clientX - lastPanX; const dy = event.clientY - lastPanY; translateX += dx; translateY += dy; lastPanX = event.clientX; lastPanY = event.clientY; redrawCanvasResults(); if (plotTooltip) plotTooltip.style.display = 'none'; } else if (baseImage && simData.hasInteractiveData) { updateTooltip(mouseX, mouseY); } });
        interactivePlotCanvas.addEventListener('mouseup', (event) => { if (event.button === 0) { isPanning = false; interactivePlotArea.classList.remove('panning'); } });
        interactivePlotCanvas.addEventListener('mouseleave', () => { isPanning = false; interactivePlotArea.classList.remove('panning'); if (plotTooltip) plotTooltip.style.display = 'none'; });
        if (resetViewButton) { resetViewButton.addEventListener('click', resetViewResults); }
    }

    // Resize Observers
    let resizeObserverResults;
    if (typeof ResizeObserver !== 'undefined' && interactivePlotArea && interactivePlotCanvas) {
        resizeObserverResults = new ResizeObserver(entries => { window.requestAnimationFrame(() => { if (!entries || !entries.length) return; const entry = entries[0]; const { width, height } = entry.contentRect; if (interactivePlotCanvas.width !== Math.round(width) || interactivePlotCanvas.height !== Math.round(height)) { setCanvasSizeResults(); } }); });
        resizeObserverResults.observe(interactivePlotArea);
    } else { let resizeTimeoutCanvas; window.addEventListener('resize', () => { clearTimeout(resizeTimeoutCanvas); resizeTimeoutCanvas = setTimeout(setCanvasSizeResults, 150); }); }

    let resizeTimeoutPlacement; window.addEventListener('resize', () => { clearTimeout(resizeTimeoutPlacement); resizeTimeoutPlacement = setTimeout(updatePlacementAreaAppearance, 150); });

    // Resize Observer para el canvas de sección transversal
    let resizeObserverCrossSection;
    if (typeof ResizeObserver !== 'undefined' && crossSectionCanvas && crossSectionCanvas.parentElement) {
        resizeObserverCrossSection = new ResizeObserver(() => {
            window.requestAnimationFrame(drawHeatsinkCrossSection);
        });
        resizeObserverCrossSection.observe(crossSectionCanvas.parentElement);
    } else {
        window.addEventListener('resize', () => {
            setTimeout(drawHeatsinkCrossSection, 150);
        });
    }


    // Eventos del Modal PDF
    if (viewPdfButton && pdfModal && pdfIframe && closePdfModalButton) { viewPdfButton.addEventListener('click', openPdfModal); closePdfModalButton.addEventListener('click', closePdfModal); pdfModal.addEventListener('click', (event) => { if (event.target === pdfModal) closePdfModal(); }); document.addEventListener('keydown', (event) => { if (event.key === 'Escape' && pdfModal.style.display === 'block') closePdfModal(); }); }

    // Eventos del Formulario Principal
        // Evento Submit del Formulario Principal
    if (mainForm) {
        mainForm.addEventListener('submit', async (event) => {
            event.preventDefault();

            // +++ CORRECCIÓN: Limpiar los datos de la simulación anterior al INICIO de una nueva +++
            lastSuccessfulSimData = null;
            resetResultDisplays(); // Ahora esta función solo limpia el DOM

            if (modules.length === 0) {
                showStatus('Error: No hay módulos añadidos.', true);
                if (resultsSummaryDiv) resultsSummaryDiv.innerHTML = '<p class="status-error-text">Añada al menos un módulo.</p>';
                showPlaceholders();
                return;
            }
            setLoadingState(true);
            if (resultsSummaryDiv) resultsSummaryDiv.innerHTML = '<p>Iniciando simulación...</p>';
            showPlaceholders();

            const currentHeatsinkParams = {
                lx: lxInput?.value, ly: lyInput?.value, t: tInput?.value,
                k_base: kBaseInput?.value,
                h_fin: hFinInput?.value, t_fin: tFinInput?.value, num_fins: numFinsInput?.value,
                w_hollow: wHollowInput?.value, h_hollow: hHollowInput?.value, num_hollow_per_fin: numHollowPerFinInput?.value,
                // Add new manual Rth parameters
                rth_heatsink_manual: rthManualInput?.value,
                use_manual_rth: useManualRthCheckbox?.checked
            };
            const currentEnvironmentParams = { t_ambient_inlet: tAmbientInput?.value, Q_total_m3_h: qAirInput?.value };
            const currentModuleDefinitions = []; const currentPowers = {};
            let errorMessages = [];

            // Validation for existing parameters (excluding rth_heatsink_manual for now)
            const paramsToValidate = {
                "Lx": currentHeatsinkParams.lx, "Ly": currentHeatsinkParams.ly,
                "t_base": currentHeatsinkParams.t, "k_base": currentHeatsinkParams.k_base,
                "T_in": currentEnvironmentParams.t_ambient_inlet, "Q_air": currentEnvironmentParams.Q_total_m3_h,
                "h_fin": currentHeatsinkParams.h_fin, "t_fin": currentHeatsinkParams.t_fin,
                "N_fins": currentHeatsinkParams.num_fins,
                "w_hollow": currentHeatsinkParams.w_hollow, "h_hollow": currentHeatsinkParams.h_hollow,
                "N_hpf": currentHeatsinkParams.num_hollow_per_fin
            };

            for (const key in paramsToValidate) {
                const valStr = paramsToValidate[key];
                // Allow num_fins, h_fin, t_fin, etc. to be empty or zero if not used,
                // but if they have a value, it must be valid.
                if (valStr === null || valStr === undefined || valStr.trim() === '') {
                    // For parameters like h_fin, t_fin, num_fins, w_hollow, h_hollow, N_hpf,
                    // an empty string might be acceptable if they are effectively zero.
                    // Lx, Ly, t_base, k_base, Rth_hs_fallback, T_in, Q_air are generally required.
                    if (["Lx", "Ly", "t_base", "k_base", "Rth_hs_fallback", "T_in", "Q_air"].includes(key)) {
                        errorMessages.push(`${key} (requerido)`);
                    }
                    // For optional fin params, if empty, they are often treated as 0 by backend or default.
                    // No error needed if empty, unless specific logic downstream requires them.
                    continue;
                }

                const val = parseFloat(valStr);
                if (isNaN(val)) {
                    errorMessages.push(`${key} (no numérico)`);
                    continue;
                }

                if (key === "Q_air" && val <= 1e-9) { errorMessages.push(`${key} (>0)`); }
                else if ((key === "Lx" || key === "Ly" || key === "t_base" || key === "k_base" || key === "Rth_hs_fallback") && val <= 1e-9) { errorMessages.push(`${key} (>0)`); }
                else if ((key === "h_fin" || key === "t_fin" || key === "w_hollow" || key === "h_hollow") && val < 0) { errorMessages.push(`${key} (>=0)`); }
                else if ((key === "N_fins" || key === "N_hpf") && (val < 0 || !Number.isInteger(val))) { errorMessages.push(`${key} (entero >=0)`); }
            }

            // Specific validation for manual Rth if checkbox is checked
            if (currentHeatsinkParams.use_manual_rth) {
                const manualRthValStr = currentHeatsinkParams.rth_heatsink_manual;
                if (manualRthValStr === null || manualRthValStr === undefined || manualRthValStr.trim() === '') {
                    errorMessages.push("Manual Rth (requerido si 'Usar Manual Rth' está activado)");
                } else {
                    const manualRthVal = parseFloat(manualRthValStr);
                    if (isNaN(manualRthVal)) {
                        errorMessages.push("Manual Rth (no numérico)");
                    } else if (manualRthVal <= 1e-9) {
                        errorMessages.push("Manual Rth (>0)");
                    }
                }
            }


            const numFinsVal = parseInt(currentHeatsinkParams.num_fins); const tFinVal = parseFloat(currentHeatsinkParams.t_fin); const hFinVal = parseFloat(currentHeatsinkParams.h_fin);
            const numHollowPerFinVal = parseInt(currentHeatsinkParams.num_hollow_per_fin); const wHollowVal = parseFloat(currentHeatsinkParams.w_hollow); const hHollowVal = parseFloat(currentHeatsinkParams.h_hollow);

            if (numFinsVal > 0 && (isNaN(tFinVal) || tFinVal <=0 || isNaN(hFinVal) || hFinVal <= 0 )) { errorMessages.push("Si N_fins > 0, t_fin y h_fin deben ser > 0"); }
            if (numHollowPerFinVal > 0) { if (numFinsVal <= 0) errorMessages.push("N_hpf > 0 requiere N_fins > 0"); if (isNaN(wHollowVal) || wHollowVal <= 0) errorMessages.push("N_hpf > 0 requiere w_hollow > 0"); if (isNaN(hHollowVal) || hHollowVal <= 0) errorMessages.push("N_hpf > 0 requiere h_hollow > 0"); if (wHollowVal >= tFinVal || hHollowVal >= hFinVal) errorMessages.push("Hueco no cabe en aleta"); }
            modules.forEach(module => { currentModuleDefinitions.push({ id: module.id, center_x: module.x, center_y: module.y }); for (const chipSuffix in module.powers) { const powerVal = module.powers[chipSuffix]; const inputElem = module.controlElement?.querySelector(`input[data-chip-suffix="${chipSuffix}"]`); if (isNaN(powerVal) || powerVal < 0) { errorMessages.push(`P(${module.id}_${chipSuffix})`); if(inputElem) inputElem.classList.add('input-error'); } else { if (inputElem) inputElem.classList.remove('input-error'); } currentPowers[`${module.id}_${chipSuffix}`] = (powerVal >= 0 ? powerVal : 0).toString(); } });

            if (errorMessages.length > 0) {
                showStatus(`Error: Campos inválidos (${errorMessages.join(', ')})`, true);
                if (resultsSummaryDiv) resultsSummaryDiv.innerHTML = `<p class="status-error-text">Corrija los errores en los controles y reintente.</p>`;
                setLoadingState(false);
                showPlaceholders();
                return;
            }
            const dataToSend = { heatsink_params: currentHeatsinkParams, environment_params: currentEnvironmentParams, powers: currentPowers, module_definitions: currentModuleDefinitions };

            try {
                const response = await fetch('/update_simulation', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(dataToSend)
                });
                const results = await response.json();

                if (!response.ok) {
                    let errorMsg = `Error Servidor: ${response.status}`;
                    if (results && results.message) errorMsg = results.message;
                    throw new Error(errorMsg);
                }

                // Guardar los datos para el PDF (lógica sin cambios, ahora funciona correctamente)
                if (results.status && results.status.startsWith('Success') && results.simulation_inputs) {
                    lastSuccessfulSimData = {
                        inputs: results.simulation_inputs,
                        outputs: results
                    };
                } else {
                    if (results.status && results.status.startsWith('Success')) {
                         console.warn("La simulación tuvo éxito, pero el backend no devolvió 'simulation_inputs'. El informe PDF no estará disponible.");
                    }
                }

                displayResults(results); // Esta llamada ya no borrará los datos.

            } catch (error) {
                console.error('Error en Fetch:', error);
                showStatus(`Error comunicación: ${error.message}`, true);
                if (resultsSummaryDiv) resultsSummaryDiv.innerHTML = `<p class="status-error-text">Error al obtener resultados: ${error.message}</p>`;
                showPlaceholders();
            } finally {
                setLoadingState(false);
            }
        });
    }

    // +++ INICIO: NUEVO MANEJADOR DE EVENTOS PARA EL BOTÓN DE PDF +++
    if (generatePdfButton) {
        generatePdfButton.addEventListener('click', async () => {
            if (!lastSuccessfulSimData) {
                showStatus("Error: No hay datos de simulación válidos para generar un informe.", true);
                return;
            }

            showStatus("Generando informe PDF...", false, true); // Muestra estado de carga
            setLoadingState(true);

            try {
                const response = await fetch('/generate_report', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(lastSuccessfulSimData) // Enviar los datos completos (inputs + outputs)
                });

                if (!response.ok) {
                    const errorJson = await response.json();
                    throw new Error(errorJson.message || `Server error: ${response.status}`);
                }

                // Recibir el PDF como un blob y crear un enlace de descarga
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.style.display = 'none';
                a.href = url;
                a.download = 'thermal_report.pdf'; // Nombre del archivo
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url); // Liberar la URL del objeto
                a.remove();

                showStatus("Informe PDF descargado con éxito.", false);

            } catch (error) {
                console.error('Error al generar PDF:', error);
                showStatus(`Error al generar el informe: ${error.message}`, true);
            } finally {
                setLoadingState(false);
            }
        });
    }
    // +++ FIN: NUEVO MANEJADOR DE EVENTOS PARA EL BOTÓN DE PDF +++


    // --- Funciones de UI (Resultados) ---
    function showPlaceholders() {
        if (interactivePlotCanvas) interactivePlotCanvas.style.display = 'none';
        if (interactivePlotPlaceholder) {
            interactivePlotPlaceholder.style.display = 'block';
            interactivePlotPlaceholder.textContent = "The interactive map will appear here.";
        }
        if (resetViewButton) resetViewButton.style.display = 'none';
        if (combinedPlotImg) combinedPlotImg.style.display = 'none';
        if (combinedPlotPlaceholder) {
            combinedPlotPlaceholder.style.display = 'block';
            combinedPlotPlaceholder.textContent = "The combined Base/Air plot will appear here.";
        }
        if (zoomPlotImg) zoomPlotImg.style.display = 'none';
        if (zoomPlotPlaceholder) {
            zoomPlotPlaceholder.style.display = 'block';
            zoomPlotPlaceholder.textContent = "The module detail plot will appear here.";
        }
    }

        function resetResultDisplays() {
        if(resultsSummaryDiv) resultsSummaryDiv.innerHTML = '<p>Waiting for new simulation...</p>';
        showPlaceholders();
        baseImage = null;
        simData = { temperature_matrix: null, x_coordinates: null, y_coordinates: null, sim_lx: null, sim_ly: null, sim_nx: null, sim_ny: null, hasInteractiveData: false };
        resetViewResults();
        if (combinedPlotImg) combinedPlotImg.src = '';
        if (zoomPlotImg) zoomPlotImg.src = '';
        if (plotTooltip) plotTooltip.style.display = 'none';
        if(interactivePlotArea) interactivePlotArea.style.cursor = 'grab';

        // OCULTAR BOTÓN DE PDF AL RESETEAR (PERO NO BORRAR DATOS)
        if (generatePdfButton) generatePdfButton.style.display = 'none';
        // LA LÍNEA "lastSuccessfulSimData = null;" HA SIDO ELIMINADA DE AQUÍ
    }

    function displayResults(results) {
         if (!resultsSummaryDiv || !interactivePlotCanvas || !interactivePlotPlaceholder || !interactivePlotArea || !combinedPlotImg || !combinedPlotPlaceholder || !zoomPlotImg || !zoomPlotPlaceholder || !ctxResults || !resetViewButton) { console.error("displayResults: Faltan elementos DOM."); showStatus("Error interno al mostrar resultados.", true); return; }
         resetResultDisplays();
         let summaryHtml = `<h3>Summary</h3>`;
         if (results.status === 'Error' || !results.status || results.status.startsWith('Processing')) {
             const errorMsg = results.error_message || 'Unknown error or simulation not completed.';
             summaryHtml += `<p class="status-error-text">Simulation Error: ${errorMsg}</p>`;
             showStatus(`Error: ${errorMsg}`, true);
             showPlaceholders();
             if(interactivePlotPlaceholder) interactivePlotPlaceholder.textContent = "Simulation error.";
         }
         else {
             // +++ MOSTRAR BOTÓN DE PDF SI HAY ÉXITO +++
             if (generatePdfButton) generatePdfButton.style.display = 'block';

             showStatus("Simulation completed.", false);
             summaryHtml += `<p>Convergence: <span class="result-value ${results.convergence ? 'status-success-text' : 'status-error-text'}">${results.convergence ? 'Yes' : 'No'}</span> (Iter.: <span class="result-value">${results.iterations ?? 'N/A'}</span>)</p><p>T Max Base: <span class="result-value">${formatTemp(results.t_max_base)}</span> °C</p><p>T Mean Base: <span class="result-value">${formatTemp(results.t_avg_base)}</span> °C</p><p>T Air Outlet: <span class="result-value">${formatTemp(results.t_air_outlet)}</span> °C</p><p>T Max Junction: <span class="result-value">${formatTemp(results.t_max_junction)}</span> °C (${results.t_max_junction_chip || 'N/A'})</p><p>T Max NTC: <span class="result-value">${formatTemp(results.t_max_ntc)}</span> °C</p>${results.simulation_time_s !== undefined ? `<p>Sim. Time: <span class="result-value">${results.simulation_time_s.toFixed(2)}</span> s</p>` : ''}`;
             summaryHtml += `<h3>Results by module</h3><div id="module-results">`;
             if (results.module_results?.length > 0) {
                 results.module_results.forEach(mod => {
                     const modId = mod?.id || '?';
                     const tNtc = formatTemp(mod?.t_ntc);
                     summaryHtml += `<div><b>${modId}:</b> T_NTC≈${tNtc}°C<br/>`;
                     if (mod?.chips?.length > 0) {
                         summaryHtml += '<ul>';
                         mod.chips.forEach(chip => {
                             const chipSuffix = chip?.suffix || '?';
                             const tBase_hs = formatTemp(chip?.t_base_heatsink);
                             const tJ = formatTemp(chip?.tj);
                             const tBase_mod = formatTemp(chip?.t_base_module_surface);
                             summaryHtml += `<li>${chipSuffix}: T<sub>base_hs</sub>≈${tBase_hs}°C, T<sub>base_mod</sub>≈${tBase_mod}°C, T<sub>j</sub>=${tJ}°C</li>`;
                         });
                         summaryHtml += '</ul>';
                     } else {
                         summaryHtml += '<span class="details-na">(No chip data)</span>';
                     }
                     summaryHtml += `</div>`;
                 });
             } else if (modules.length > 0) {
                 summaryHtml += '<p>No detailed module results received.</p>';
             } else {
                 summaryHtml += '<p>Simulation performed without modules.</p>';
             }
             summaryHtml += `</div>`;
             const hasNumericalData = results.temperature_matrix && Array.isArray(results.temperature_matrix) && results.x_coordinates && Array.isArray(results.x_coordinates) && results.y_coordinates && Array.isArray(results.y_coordinates) && results.sim_lx != null && results.sim_ly != null && results.sim_nx != null && results.sim_ny != null;
             if (results.plot_interactive_raw_uri) {
                 interactivePlotPlaceholder.textContent = 'Loading interactive map...';
                 interactivePlotCanvas.style.display = 'none';
                 resetViewButton.style.display = 'none';
                 baseImage = new Image();
                 baseImage.onload = () => {
                     interactivePlotCanvas.style.display = 'block';
                     interactivePlotPlaceholder.style.display = 'none';
                     resetViewButton.style.display = 'inline-block';
                     setCanvasSizeResults();
                     if (hasNumericalData) {
                         simData = { temperature_matrix: results.temperature_matrix, x_coordinates: results.x_coordinates, y_coordinates: results.y_coordinates, sim_lx: results.sim_lx, sim_ly: results.sim_ly, sim_nx: results.sim_nx, sim_ny: results.sim_ny, hasInteractiveData: true };
                         interactivePlotArea.style.cursor = 'crosshair';
                         showStatus("Simulation and interactive map loaded.", false);
                     } else {
                         simData.hasInteractiveData = false;
                         interactivePlotArea.style.cursor = 'grab';
                         if (results.status === 'Success' || results.status === 'Success_NoInteractiveData') {
                             showStatus("Simulation completed (interactive map data missing).", false);
                         }
                     }
                 };
                 baseImage.onerror = () => {
                     showPlaceholders();
                     interactivePlotPlaceholder.textContent = "Error loading interactive map.";
                     if(resetViewButton) resetViewButton.style.display = 'none';
                     showStatus("Error loading map image.", true);
                     simData.hasInteractiveData = false;
                 };
                 baseImage.src = results.plot_interactive_raw_uri;
             }
             else {
                 interactivePlotCanvas.style.display = 'none';
                 interactivePlotPlaceholder.textContent = "Interactive map not available.";
                 if(resetViewButton) resetViewButton.style.display = 'none';
                 interactivePlotArea.style.cursor = 'default';
                 simData.hasInteractiveData = false;
                 if (results.status === 'Success' || results.status === 'Success_NoInteractiveData') {
                     showStatus("Simulation completed (interactive map not available).", false);
                 }
             }
             if (results.plot_base_data_uri) {
                 combinedPlotImg.src = results.plot_base_data_uri;
                 combinedPlotImg.style.display = 'block';
                 combinedPlotPlaceholder.style.display = 'none';
             } else {
                 combinedPlotImg.style.display = 'none';
                 combinedPlotPlaceholder.style.display = 'block';
                 combinedPlotPlaceholder.textContent = "Combined Base/Air plot not available.";
             }
             if (results.plot_zoom_data_uri) {
                 zoomPlotImg.src = results.plot_zoom_data_uri;
                 zoomPlotImg.style.display = 'block';
                 zoomPlotPlaceholder.style.display = 'none';
             } else {
                 zoomPlotImg.style.display = 'none';
                 zoomPlotPlaceholder.style.display = 'block';
                 zoomPlotPlaceholder.textContent = "Module detail plot not available.";
             }
         }
         if(resultsSummaryDiv) resultsSummaryDiv.innerHTML = summaryHtml;
    }
    function formatTemp(value, precision = 1) { /* ... (sin cambios) ... */
        const num = parseFloat(value);
        if (value === null || value === undefined || isNaN(num)) {
            return 'N/A';
        }
        return num.toFixed(precision);
    }
    function setLoadingState(isLoading) { /* ... (sin cambios) ... */
        if (isLoading) {
            showStatus('Calculating...', false, true);
            if (loader) loader.style.display = 'inline-block';
            if (updateButton) updateButton.disabled = true;
            document.body.classList.add('loading');
        } else {
            if (loader) loader.style.display = 'none';
            if (updateButton) updateButton.disabled = false;
            document.body.classList.remove('loading');
        }
    }
    function showStatus(message, isError = false, isLoading = false) { /* ... (sin cambios) ... */
        if (!statusDiv) return;
        statusDiv.textContent = message;
        statusDiv.className = 'status-neutral';
        if (isLoading) {
            statusDiv.classList.add('status-loading');
        } else if (isError) {
            statusDiv.classList.add('status-error');
        } else {
            statusDiv.classList.add('status-success');
        }
    }

    // --- Inicialización ---
    updatePlacementAreaAppearance(); // Dibuja aletas en placement y prepara el área
    drawHeatsinkCrossSection();    // Dibuja la sección transversal inicial
    if (noModulesMessage) noModulesMessage.style.display = modules.length === 0 ? 'block' : 'none';
    resetResultDisplays();
    if(resultsSummaryDiv) resultsSummaryDiv.innerHTML = '<p>Add modules, adjust parameters, and click "Update Simulation".</p>';
    showStatus("Ready to simulate", false);
    setCanvasSizeResults(); // Prepara canvas de resultados
    console.log(">>> Inicialización completada.");
});
