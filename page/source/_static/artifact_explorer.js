// AI (ChatGPT) GENERATED CODE

d3.json(window.ARTIFACTS_JSON).then(data => {
    // ---------- Configuration ----------
    const container = d3.select("#artifact-graph");
    container.selectAll("*").remove();

    const containerNode = container.node();
    const containerWidth = Math.max(600, containerNode.clientWidth || 900);
    const containerHeight = Math.max(400, containerNode.clientHeight || 700);

    const css = getComputedStyle(containerNode);
    const colorText = css.getPropertyValue("--pst-color-text").trim() || css.getPropertyValue("color").trim() || "#222";
    const colorMuted = css.getPropertyValue("--pst-color-text-muted").trim() || "#666";
    const bg = css.getPropertyValue("background-color").trim() || "#fff";

    function isDarkBackground(bgColor) {
        try {
            if (bgColor.startsWith("rgb")) {
                const nums = bgColor.match(/[\d.]+\b/g).slice(0, 3).map(Number);
                const [r, g, b] = nums.map(v => v / 255);
                const lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
                return lum < 0.5;
            } else if (bgColor.startsWith("#")) {
                const hex = bgColor.slice(1);
                const bigint = parseInt(hex.length === 3 ? hex.split('').map(c => c + c).join('') : hex, 16);
                const r = ((bigint >> 16) & 255) / 255,
                    g = ((bigint >> 8) & 255) / 255,
                    b = (bigint & 255) / 255;
                const lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
                return lum < 0.5;
            }
        } catch (_) { }
        return false;
    }

    const darkMode = isDarkBackground(bg);
    const edgeColorVersion = darkMode ? "#ddd" : "#222";
    const edgeColorDep = darkMode ? "rgba(220,220,220,0.6)" : "rgba(80,80,80,0.5)";
    const timelineColor = colorMuted || (darkMode ? "#999" : "#777");
    const textColor = colorText || (darkMode ? "#eee" : "#111");
    const nodeStroke = darkMode ? "#ddd" : "#222";

    const typeStyles = {
        code_version: { color: "#2b6cb0", shape: "rect", column: 0, label: "Code Version" },
        docker_image: { color: "#d97706", shape: "circle", column: 1, label: "Docker Image" },
        dataset: { color: "#059669", shape: "diamond", column: 2, label: "Dataset" },
        experiment_data: { color: "#7c3aed", shape: "triangle", column: 3, label: "Experiment Data" }
    };

    // ---------- Data ----------
    const artifacts = data.artifacts.map(a => {
        const [day, mon, year] = (a.date || "01/01/1970").split("/").map(Number);
        const dateObj = new Date(year, mon - 1, day);
        return { ...a, dateObj };
    });

    artifacts.sort((a, b) => a.dateObj - b.dateObj || a.name.localeCompare(b.name));

    const nodes = artifacts.map(a => ({
        id: a.name,
        type: a.type,
        url: a.url || null,
        dateObj: a.dateObj,
        raw: a
    }));

    const nodeMap = new Map(nodes.map(n => [n.id, n]));

    const links = [];
    artifacts.forEach(a => {
        (a.dependencies || []).forEach(dep => {
            if (nodeMap.has(dep)) links.push({ source: dep, target: a.name, kind: "dep" });
        });
        if (a.previous_version && nodeMap.has(a.previous_version)) {
            links.push({ source: a.previous_version, target: a.name, kind: "version" });
        }
    });

    // ---------- Layout ----------
    const dateKeys = [...new Set(nodes.map(n => n.dateObj.getTime()))].sort();
    const yScale = d3.scalePoint()
        .domain(dateKeys.map(String))
        .range([100, containerHeight - 100])
        .padding(0.6);

    const nodesByDate = d3.group(nodes, n => String(n.dateObj.getTime()));
    const columnIndices = [0, 1, 2, 3];

    const columnX = d3.scaleBand()
        .domain(columnIndices)
        .range([200, Math.max(600, containerWidth) - 100])
        .padding(0.45);

    nodes.forEach(n => {
        const dateKey = String(n.dateObj.getTime());
        const group = nodesByDate.get(dateKey);
        const indexInGroup = group.indexOf(n);
        const groupSize = group.length;

        const centralY = yScale(dateKey);
        const spread = Math.min(20 + groupSize * 4, 60);
        const offset = (indexInGroup - (groupSize - 1) / 2) * (spread / Math.max(1, groupSize - 1));

        n.y = centralY + offset;

        const ts = typeStyles[n.type] || { column: 3 };
        n.column = ts.column ?? 3;

        const jitter = (indexInGroup % 3) * 6 - 6;
        n.x = columnX(n.column) + jitter;
    });

    // ---------- enforce minimal vertical spacing per column ----------
    const MIN_Y_GAP = 28;
    const nodesByColumn = d3.group(nodes, n => n.column);
    nodesByColumn.forEach(colNodes => {
        colNodes.sort((a, b) => a.y - b.y);
        for (let i = 1; i < colNodes.length; i++) {
            const prev = colNodes[i - 1];
            const curr = colNodes[i];
            if (curr.y - prev.y < MIN_Y_GAP) {
                curr.y = prev.y + MIN_Y_GAP;
            }
        }
    });

    // ---------- SVG / Zoom ----------
    const svg = container.append("svg")
        .attr("width", containerWidth)
        .attr("height", containerHeight)
        .attr("viewBox", `0 0 ${containerWidth} ${containerHeight}`);

    const content = svg.append("g").attr("class", "content");

    svg.call(d3.zoom().scaleExtent([0.5, 3]).on("zoom", ev => content.attr("transform", ev.transform)));

    // ---------- Timeline ----------
    content.append("line")
        .attr("x1", 120).attr("y1", 40)
        .attr("x2", 120).attr("y2", containerHeight - 40)
        .attr("stroke", timelineColor)
        .attr("stroke-width", 2);

    const dateLabelGroup = [...nodesByDate.entries()].map(([dateKey]) => ({
        dateKey,
        centralY: yScale(dateKey),
        label: (new Date(Number(dateKey))).toLocaleDateString("pt-BR")
    }));

    content.selectAll(".date-label")
        .data(dateLabelGroup)
        .enter()
        .append("text")
        .attr("x", 100)
        .attr("y", d => d.centralY)
        .attr("dy", "0.35em")
        .attr("text-anchor", "end")
        .style("font-size", "12px")
        .style("fill", colorMuted)
        .text(d => d.label);

    // ---------- Links ----------
    const defs = svg.append("defs");

    defs.append("marker")
        .attr("id", "arrow-version")
        .attr("viewBox", "0 0 10 10")
        .attr("refX", 24)
        .attr("refY", 5)
        .attr("markerWidth", 9)
        .attr("markerHeight", 9)
        .attr("orient", "auto")
        .append("path")
        .attr("d", "M 0 0 L 10 5 L 0 10 z")
        .attr("fill", edgeColorVersion);

    defs.append("marker")
        .attr("id", "arrow-dep")
        .attr("viewBox", "0 0 10 10")
        .attr("refX", 22)
        .attr("refY", 5)
        .attr("markerWidth", 8)
        .attr("markerHeight", 8)
        .attr("orient", "auto")
        .append("path")
        .attr("d", "M 0 0 L 10 5 L 0 10 z")
        .attr("fill", edgeColorDep);

    function orthPath(sx, sy, tx, ty) {
        const midX = sx + (tx - sx) * 0.45;
        return `M ${sx} ${sy} L ${midX} ${sy} L ${midX} ${ty} L ${tx} ${ty}`;
    }

    const linkG = content.append("g");
    const edgeKeyCounts = {};
    links.forEach(l => {
        const key = `${l.source}=>${l.target}`;
        edgeKeyCounts[key] = (edgeKeyCounts[key] || 0) + 1;
    });

    linkG.selectAll("path")
        .data(links)
        .enter()
        .append("path")
        .attr("fill", "none")
        .attr("stroke", d => d.kind === "version" ? edgeColorVersion : edgeColorDep)
        .attr("stroke-width", d => d.kind === "version" ? 2.4 : 1.6)
        .attr("stroke-dasharray", d => d.kind === "dep" ? "4 3" : "0")
        .attr("marker-end", d => d.kind === "version" ? "url(#arrow-version)" : "url(#arrow-dep)")
        .attr("opacity", 0.95)
        .attr("d", function (d) {
            const s = nodeMap.get(d.source);
            const t = nodeMap.get(d.target);

            const key = `${d.source}=>${d.target}`;
            const count = edgeKeyCounts[key];

            if (count === 1) return orthPath(s.x, s.y, t.x, t.y);

            const same = links.filter(x => `${x.source}=>${x.target}` === key);
            const index = same.indexOf(d);

            const baseMid = s.x + (t.x - s.x) * 0.45;
            const spread = 8 * (index - (count - 1) / 2);
            const midX = baseMid + spread;

            return `M ${s.x} ${s.y} L ${midX} ${s.y} L ${midX} ${t.y} L ${t.x} ${t.y}`;
        });

    // ---------- Tooltip ----------
    const tooltip = d3.select("body")
        .append("div")
        .attr("class", "artifact-tooltip")
        .style("position", "absolute")
        .style("padding", "6px 10px")
        .style("background", "rgba(0,0,0,0.75)")
        .style("color", "white")
        .style("border-radius", "6px")
        .style("font-size", "13px")
        .style("pointer-events", "none")
        .style("opacity", 0);

    // ---------- Nodes ----------
    const nodeG = content.selectAll("g.node")
        .data(nodes)
        .enter()
        .append("g")
        .attr("class", "node")
        .attr("transform", d => `translate(${d.x},${d.y})`)
        .style("cursor", d => d.url ? "pointer" : "default")
        .on("click", (_, d) => d.url && window.open(d.url, "_blank"));

    nodeG.each(function (d) {
        const s = typeStyles[d.type] || { shape: "circle", color: "#888" };
        const g = d3.select(this);

        if (s.shape === "circle") g.append("circle").attr("r", 18).attr("fill", s.color).attr("stroke", nodeStroke).attr("stroke-width", 1.5);
        else if (s.shape === "rect") g.append("rect").attr("x", -22).attr("y", -12).attr("width", 44).attr("height", 24).attr("rx", 4).attr("fill", s.color).attr("stroke", nodeStroke).attr("stroke-width", 1.5);
        else if (s.shape === "diamond") g.append("path").attr("d", "M 0 -16 L 14 0 L 0 16 L -14 0 Z").attr("fill", s.color).attr("stroke", nodeStroke).attr("stroke-width", 1.2);
        else if (s.shape === "triangle") g.append("path").attr("d", "M -16 10 L 16 10 L 0 -14 Z").attr("fill", s.color).attr("stroke", nodeStroke).attr("stroke-width", 1.2);
    });

    nodeG.each(function (d) {
        const g = d3.select(this);
        const text = g.append("text").text(d.id).attr("x", 30).attr("dy", "0.35em").style("font-size", "12px").style("fill", textColor);
        const bbox = text.node().getBBox();
        g.insert("rect", "text")
            .attr("x", bbox.x - 4).attr("y", bbox.y - 2)
            .attr("width", bbox.width + 8).attr("height", bbox.height + 4)
            .attr("rx", 4)
            .attr("fill", darkMode ? "rgba(0,0,0,0.45)" : "rgba(255,255,255,0.75)")
            .attr("stroke", darkMode ? "rgba(255,255,255,0.25)" : "rgba(0,0,0,0.25)")
            .attr("stroke-width", 0.5);
    });

    // ---------- Attach tooltip ----------
    nodeG
        .on("mouseenter", (event, d) => {
            tooltip
                .style("opacity", 1)
                .html(`
                    <strong>${d.id}</strong><br>
                    <em>${typeStyles[d.type]?.label || d.type}</em><br>
                    <small>${d.raw.date || ""}</small><br>
                    <div style="margin-top:6px; white-space: normal; max-width: 300px; word-wrap: break-word;">
                        ${d.raw.description || ""}
                    </div>
                `);
        })
        .on("mousemove", event => {
            tooltip.style("top", (event.pageY + 12) + "px")
                .style("left", (event.pageX + 12) + "px");
        })
        .on("mouseleave", () => tooltip.style("opacity", 0));

    // ---------- Legend ----------
    const legend = svg.append("g").attr("transform", `translate(${containerWidth - 240}, 40)`);
    legend.append("rect")
        .attr("x", -12).attr("y", -28).attr("width", 230)
        .attr("height", Object.keys(typeStyles).length * 28 + 44)
        .attr("rx", 6)
        .attr("fill", darkMode ? "rgba(255,255,255,0.03)" : "rgba(0,0,0,0.03)")
        .attr("stroke", timelineColor)
        .attr("stroke-width", 0.6);
    legend.append("text").attr("x", 0).attr("y", -10)
        .text("Artifact Type").style("font-size", "13px").style("font-weight", "600").style("fill", textColor);

    Object.entries(typeStyles).forEach(([key, s], i) => {
        const g = legend.append("g").attr("transform", `translate(8, ${i * 28 + 10})`);
        if (s.shape === "circle") g.append("circle").attr("r", 8).attr("cx", 8).attr("cy", 8).attr("fill", s.color);
        else if (s.shape === "rect") g.append("rect").attr("x", 0).attr("y", 0).attr("width", 18).attr("height", 14).attr("rx", 3).attr("fill", s.color);
        else g.append("path").attr("d", s.shape === "diamond" ? "M 9 0 L 18 9 L 9 18 L 0 9 Z" : "M 0 14 L 18 14 L 9 0 Z").attr("fill", s.color);
        g.append("text").attr("x", 36).attr("y", 12).text(s.label).style("font-size", "12px").style("fill", textColor);
    });

    // ---------- Resize ----------
    let resizeTimer = null;
    window.addEventListener("resize", () => {
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(() => {
            const w = Math.max(600, container.node().clientWidth || containerWidth);
            const h = Math.max(400, container.node().clientHeight || containerHeight);
            svg.attr("width", w).attr("height", h).attr("viewBox", `0 0 ${w} ${h}`);
        }, 150);
    });
});
