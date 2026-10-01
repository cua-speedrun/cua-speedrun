/* Piecewise-linear, dominance-preserving interpolation of measured frontiers.
 * PAINT-inspired pruning (Hartikainen, Miettinen & Wiecek, 2012): check whole
 * simplices against themselves, every input point and every retained simplex.
 * Candidates are faces of the 3D Delaunay tessellation. Coplanar inputs use
 * a 2D projection perpendicular to the all-objectives-worse direction.
 * No measured values are changed. Geometry is normalized before this module.
 */
(function (root) {
  "use strict";
  const FEASIBILITY_EPS = 1e-10;
  const DOMINANCE_EPS = 1e-8;
  const objectives = (p) => [p.cost, p.time, -p.performance];

  function solve(matrix, rhs) {
    const rows = matrix.map((row, i) => [...row, rhs[i]]);
    const size = rhs.length;
    for (let col = 0; col < size; col += 1) {
      let pivot = col;
      for (let row = col + 1; row < size; row += 1) {
        if (Math.abs(rows[row][col]) > Math.abs(rows[pivot][col])) pivot = row;
      }
      if (Math.abs(rows[pivot][col]) < 1e-12) return null;
      [rows[col], rows[pivot]] = [rows[pivot], rows[col]];
      const divisor = rows[col][col];
      for (let k = col; k <= size; k += 1) rows[col][k] /= divisor;
      for (let row = 0; row < size; row += 1) {
        if (row === col) continue;
        const factor = rows[row][col];
        for (let k = col; k <= size; k += 1) rows[row][k] -= factor * rows[col][k];
      }
    }
    return rows.map((row) => row[size]);
  }

  const basisCache = new Map();
  function bases(count) {
    if (basisCache.has(count)) return basisCache.get(count);
    const result = [];
    const visit = (start, chosen) => {
      if (chosen.length === 5) { result.push(chosen); return; }
      for (let i = start; i <= count - (5 - chosen.length); i += 1) {
        visit(i + 1, [...chosen, i]);
      }
    };
    visit(0, []);
    basisCache.set(count, result);
    return result;
  }

  // Does any convex combination in A dominate one in B? Solve the bounded LP
  // max sum(slack), subject to B*beta - A*alpha = slack >= 0 and both weights
  // summing to 1. At most nine variables / five equalities for two triangles:
  // enumerate the <=126 basic solutions, not a sampling of triangle interiors.
  function dominatesPolytope(a, b) {
    const left = a.map(objectives);
    const right = b.map(objectives);
    if ([0, 1, 2].some((d) =>
      Math.min(...left.map((p) => p[d])) > Math.max(...right.map((p) => p[d])) + FEASIBILITY_EPS
    )) return false;
    const columns = [
      ...left.map((p) => [1, 0, ...p.map((v) => -v)]),
      ...right.map((p) => [0, 1, ...p]),
      [0, 0, -1, 0, 0], [0, 0, 0, -1, 0], [0, 0, 0, 0, -1],
    ];
    const firstSlack = a.length + b.length;
    for (const basis of bases(columns.length)) {
      const matrix = [0, 1, 2, 3, 4].map((row) => basis.map((col) => columns[col][row]));
      const values = solve(matrix, [1, 1, 0, 0, 0]);
      if (!values || values.some((v) => !Number.isFinite(v) || v < -FEASIBILITY_EPS)) continue;
      const improvement = values.reduce((sum, value, i) => sum + (basis[i] >= firstSlack ? value : 0), 0);
      if (improvement > DOMINANCE_EPS) return true;
    }
    return false;
  }

  function circumcircle(a, b, c) {
    const d = 2 * (a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y));
    if (Math.abs(d) < 1e-12) return null;
    const norm = (p) => p.x * p.x + p.y * p.y;
    const x = (norm(a) * (b.y - c.y) + norm(b) * (c.y - a.y) + norm(c) * (a.y - b.y)) / d;
    const y = (norm(a) * (c.x - b.x) + norm(b) * (a.x - c.x) + norm(c) * (b.x - a.x)) / d;
    return { x, y, r2: (x - a.x) ** 2 + (y - a.y) ** 2 };
  }

  // Deterministic Bowyer-Watson fallback for coplanar observations.
  function projectedTriangles(points) {
    if (points.length < 3) return [];
    const projected = points.map((p) => ({
      x: (p.cost - p.time) / Math.sqrt(2),
      y: (p.cost + p.time + 2 * p.performance) / Math.sqrt(6),
    }));
    const xs = projected.map((p) => p.x), ys = projected.map((p) => p.y);
    const cx = (Math.min(...xs) + Math.max(...xs)) / 2;
    const cy = (Math.min(...ys) + Math.max(...ys)) / 2;
    const span = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys), 1e-6);
    const n = points.length;
    projected.push({ x: cx - 32 * span, y: cy - span },
      { x: cx, y: cy + 32 * span }, { x: cx + 32 * span, y: cy - span });
    const make = (ids) => ({ ids, circle: circumcircle(...ids.map((i) => projected[i])) });
    let triangles = [make([n, n + 1, n + 2])];
    for (let i = 0; i < n; i += 1) {
      const p = projected[i];
      const boundary = new Map();
      triangles = triangles.filter((triangle) => {
        const circle = triangle.circle;
        if (!circle || (p.x - circle.x) ** 2 + (p.y - circle.y) ** 2 > circle.r2 + 1e-12) return true;
        triangle.ids.forEach((a, j) => {
          const edge = [a, triangle.ids[(j + 1) % 3]].sort((a, b) => a - b);
          const key = edge.join(":");
          if (boundary.has(key)) boundary.delete(key); else boundary.set(key, edge);
        });
        return false;
      });
      for (const edge of boundary.values()) {
        const triangle = make([...edge, i]);
        if (triangle.circle) triangles.push(triangle);
      }
    }
    return triangles.filter((t) => t.ids.every((i) => i < n)).map((t) => t.ids.sort((a, b) => a - b));
  }

  function candidateTriangles(points) {
    if (points.length < 4) return projectedTriangles(points);
    const xyz = points.map(objectives);
    const low = [0, 1, 2].map((d) => Math.min(...xyz.map((p) => p[d])));
    const high = [0, 1, 2].map((d) => Math.max(...xyz.map((p) => p[d])));
    const center = low.map((v, d) => (v + high[d]) / 2);
    const span = 64 * Math.max(...high.map((v, d) => v - low[d]), 1e-6);
    const n = xyz.length;
    [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]].forEach((signs) =>
      xyz.push(center.map((v, d) => v + signs[d] * span))
    );
    const norm = (p) => p.reduce((s, v) => s + v * v, 0);
    const squaredDistance = (a, b) => a.reduce((s, v, d) => s + (v - b[d]) ** 2, 0);
    const make = (ids) => {
      const [a, ...rest] = ids.map((i) => xyz[i]);
      const center = solve(rest.map((p) => p.map((v, d) => 2 * (v - a[d]))),
        rest.map((p) => norm(p) - norm(a)));
      return center ? { ids, center, r2: squaredDistance(center, a) } : null;
    };
    const faces = (ids) => ids.map((_, i) => ids.filter((_, j) => j !== i).sort((a, b) => a - b));
    let tetrahedra = [make([n, n + 1, n + 2, n + 3])];
    for (let i = 0; i < n; i += 1) {
      const boundary = new Map();
      tetrahedra = tetrahedra.filter((tetra) => {
        if (squaredDistance(tetra.center, xyz[i]) > tetra.r2 + 1e-10) return true;
        faces(tetra.ids).forEach((face) => {
          const key = face.join(":");
          if (boundary.has(key)) boundary.delete(key); else boundary.set(key, face);
        });
        return false;
      });
      for (const face of boundary.values()) {
        const tetra = make([...face, i]);
        if (tetra) tetrahedra.push(tetra);
      }
    }
    const candidates = new Map();
    tetrahedra.filter((t) => t.ids.every((i) => i < n)).forEach((t) =>
      faces(t.ids).forEach((face) => candidates.set(face.join(":"), face))
    );
    // Nondegenerate tetrahedra have dominated interior pairs, so PAINT's
    // pruning necessarily discards them; only their faces/edges are candidates.
    return candidates.size ? [...candidates.values()] : projectedTriangles(points);
  }

  function build(input) {
    const points = Array.from(new Map(input.map((p) => [`${p.cost}:${p.time}:${p.performance}`, p])).values())
      .sort((a, b) => a.cost - b.cost || a.time - b.time || a.performance - b.performance);
    const candidates = candidateTriangles(points);
    const distance = (a, b) => (a.cost - b.cost) ** 2 + (a.time - b.time) ** 2 + (a.performance - b.performance) ** 2;
    const span = (ids) => Math.max(...ids.flatMap((i) => ids.map((j) => distance(points[i], points[j]))));
    candidates.sort((a, b) => span(a) - span(b) || a.join(":").localeCompare(b.join(":")));
    const retained = [];
    const stats = { candidates: candidates.length, internal: 0, observed: 0, patches: 0 };
    const accept = (ids, count = false) => {
      const patch = ids.map((i) => points[i]);
      if (dominatesPolytope(patch, patch)) { if (count) stats.internal += 1; return false; }
      if (points.some((p) => dominatesPolytope(patch, [p]) || dominatesPolytope([p], patch))) {
        if (count) stats.observed += 1;
        return false;
      }
      if (retained.some((other) => dominatesPolytope(patch, other) || dominatesPolytope(other, patch))) {
        if (count) stats.patches += 1;
        return false;
      }
      retained.push(patch);
      return true;
    };
    const accepted = candidates.filter((ids) => accept(ids, true));
    const triangleEdges = new Map();
    accepted.forEach((ids) => ids.forEach((a, i) => {
      const edge = [a, ids[(i + 1) % 3]].sort((a, b) => a - b);
      triangleEdges.set(edge.join(":"), edge);
    }));
    const extraEdges = new Map();
    candidates.forEach((ids) => ids.forEach((a, i) => {
      const edge = [a, ids[(i + 1) % 3]].sort((a, b) => a - b);
      if (!triangleEdges.has(edge.join(":"))) extraEdges.set(edge.join(":"), edge);
    }));
    // A collinear or two-point frontier still has a meaningful line comparison.
    if (!candidates.length) {
      for (let i = 1; i < points.length; i += 1) extraEdges.set(`${i - 1}:${i}`, [i - 1, i]);
    }
    const edges = [...extraEdges.values()].sort((a, b) => span(a) - span(b))
      .filter((ids) => accept(ids));
    return {
      points,
      triangles: accepted.map((ids) => ids.map((i) => points[i])),
      edges: [...triangleEdges.values(), ...edges].map((ids) => ids.map((i) => points[i])),
      stats,
    };
  }

  const api = { build, dominatesPolytope, candidateTriangles, tolerance: DOMINANCE_EPS };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.ParetoSurface = api;
})(typeof globalThis === "undefined" ? this : globalThis);
