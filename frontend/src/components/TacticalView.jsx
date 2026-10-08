import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Stars } from '@react-three/drei';
import * as THREE from 'three';

// Terrain values mirror disaster_sim.digital_twin.city_config.Terrain.
const TERRAIN_STYLE = {
  0: { color: '#3d6670', height: 0.08, roughness: 1 },
  1: { color: '#394955', height: 0.1, roughness: 0.78 },
  2: { color: '#7893a8', height: 4.2, roughness: 0.68 },
  3: { color: '#a85c4d', height: 1.35, roughness: 0.9 },
  4: { color: '#9d7558', height: 0.28, roughness: 1 },
  5: { color: '#18bce8', height: 0.14, roughness: 0.1, metalness: 0.45 },
  6: { color: '#663630', height: 0.13, roughness: 1 },
  7: { color: '#f0f6f6', height: 3.4, roughness: 0.5 },
  8: { color: '#27c4b9', height: 0.6, roughness: 0.32, metalness: 0.35 },
  9: { color: '#4f9663', height: 0.12, roughness: 1 },
};

// The tallest intact building is approximately 6.6 scene units. Keeping the
// drone 14 units above its current terrain gives the FPV camera roof clearance.
const DRONE_FLIGHT_ALTITUDE = 14;

// Ambulance chase-cam sits slightly above and behind the vehicle.
const AMBULANCE_CAMERA_HEIGHT = 1.6;
const AMBULANCE_CAMERA_BEHIND = 2.8;

const VICTIM_COLORS = {
  critical: '#ff5364',
  serious: '#ffad4d',
  stable: '#f5d76e',
};

function hash2D(row, col, salt = 0) {
  const value = Math.sin((row * 127.1) + (col * 311.7) + (salt * 74.7)) * 43758.5453123;
  return value - Math.floor(value);
}

function terrainHeight(type, row, col) {
  const base = TERRAIN_STYLE[type]?.height ?? 0.08;
  if (type === 2) return base + Math.floor(hash2D(Math.floor(row / 3), Math.floor(col / 3), 2) * 4) * 0.78;
  if (type === 3) return base + hash2D(row, col, 4) * 0.55;
  return base;
}

// Ground vehicles drive on roads/flat terrain and should never be placed on
// rooftops. Cap the height just above the tallest drivable surface (bridge at
// 0.6) so ambulances and robots stay at street level even when Math.round()
// momentarily maps them to a building or hospital cell.
const GROUND_VEHICLE_MAX_TERRAIN = 0.65;

function groundVehicleHeight(type, row, col) {
  return Math.min(terrainHeight(type, row, col), GROUND_VEHICLE_MAX_TERRAIN);
}

function cellToWorld(row, col, centerRow, centerCol) {
  return [col - centerCol, row - centerRow];
}

function headingFromTrail(agentId, trails) {
  const trail = trails?.[agentId] || [];
  for (let i = trail.length - 1; i > 0; i -= 1) {
    const current = trail[i];
    const previous = trail[i - 1];
    const dc = current.c - previous.c;
    const dr = current.r - previous.r;
    if (dc !== 0 || dr !== 0) return Math.atan2(dc, dr);
  }
  return 0;
}

export default function TacticalView({ worldState, agentTrails = {} }) {
  const [selectedDroneId, setSelectedDroneId] = useState(null);
  const [selectedAmbulanceId, setSelectedAmbulanceId] = useState(null);

  const drones = Object.entries(worldState?.agents || {})
    .filter(([, agent]) => agent.type === 'drone')
    .map(([id, agent]) => ({ id, ...agent }));

  const ambulances = Object.entries(worldState?.agents || {})
    .filter(([, agent]) => agent.type === 'ambulance')
    .map(([id, agent]) => ({ id, ...agent }));

  useEffect(() => {
    if (selectedDroneId && !drones.some((drone) => drone.id === selectedDroneId)) {
      setSelectedDroneId(null);
    }
  }, [drones, selectedDroneId]);

  useEffect(() => {
    if (selectedAmbulanceId && !ambulances.some((amb) => amb.id === selectedAmbulanceId)) {
      setSelectedAmbulanceId(null);
    }
  }, [ambulances, selectedAmbulanceId]);

  if (!worldState?.grid) {
    return <div className="view-loading">Establishing tactical uplink…</div>;
  }

  const selectedDrone = drones.find((drone) => drone.id === selectedDroneId);
  const selectedAmbulance = ambulances.find((amb) => amb.id === selectedAmbulanceId);
  // Only one POV can be active at a time
  const activePovId = selectedDroneId || selectedAmbulanceId;
  const activePovType = selectedDroneId ? 'drone' : selectedAmbulanceId ? 'ambulance' : null;

  const selectDrone = (id) => {
    setSelectedAmbulanceId(null);
    setSelectedDroneId(id);
  };

  const selectAmbulance = (id) => {
    setSelectedDroneId(null);
    setSelectedAmbulanceId(id);
  };

  const selectOverview = () => {
    setSelectedDroneId(null);
    setSelectedAmbulanceId(null);
  };

  return (
    <div className="tactical-shell">
      <Canvas
        dpr={[1, 1.5]}
        camera={{ position: [0, 96, 70], fov: 46, near: 0.1, far: 700 }}
        gl={{ antialias: true, toneMapping: THREE.ACESFilmicToneMapping, toneMappingExposure: 1.38 }}
      >
        <TacticalScene
          worldState={worldState}
          agentTrails={agentTrails}
          selectedDroneId={selectedDroneId}
          selectedAmbulanceId={selectedAmbulanceId}
        />
      </Canvas>

      <div className="tactical-vignette" aria-hidden="true" />
      <section className="drone-camera-panel" aria-label="Camera controls">
        <div className="panel-kicker">Live 3D camera</div>
        <div className="camera-panel-heading">
          <span className="camera-live-dot" />
          {selectedDrone
            ? `${formatDroneName(selectedDrone.id)} · FPV`
            : selectedAmbulance
              ? `${formatAmbulanceName(selectedAmbulance.id)} · Chase`
              : 'City overview'}
        </div>
        <p className="camera-panel-copy">
          {selectedDrone
            ? `Following ${selectedDrone.status || 'searching'} flight at ${(selectedDrone.battery * 100).toFixed(0)}% power · ${DRONE_FLIGHT_ALTITUDE}m clearance.`
            : selectedAmbulance
              ? `Tracking ${selectedAmbulance.status || 'idle'} unit at ${(selectedAmbulance.battery * 100).toFixed(0)}% power${selectedAmbulance.carrying_victim ? ' · PATIENT ONBOARD' : ''}.`
              : 'Choose any active unit to look through its camera.'}
        </p>
        <div className="camera-button-list">
          <button
            type="button"
            className={`camera-choice ${!activePovId ? 'is-active' : ''}`}
            onClick={selectOverview}
          >
            <span className="camera-choice-icon">◎</span>
            <span>Overview</span>
          </button>

          {drones.length > 0 && (
            <div className="camera-group-label">Aerial units</div>
          )}
          {drones.map((drone, index) => (
            <button
              type="button"
              key={drone.id}
              className={`camera-choice ${selectedDroneId === drone.id ? 'is-active' : ''}`}
              onClick={() => selectDrone(drone.id)}
            >
              <span className="camera-choice-icon">⌁</span>
              <span>{formatDroneName(drone.id, index)}</span>
              <span className="camera-choice-power">{Math.round(drone.battery * 100)}%</span>
            </button>
          ))}

          {ambulances.length > 0 && (
            <div className="camera-group-label">Ground units</div>
          )}
          {ambulances.map((amb, index) => (
            <button
              type="button"
              key={amb.id}
              className={`camera-choice ${selectedAmbulanceId === amb.id ? 'is-active' : ''}`}
              onClick={() => selectAmbulance(amb.id)}
            >
              <span className="camera-choice-icon">{amb.carrying_victim ? '🚨' : '🚑'}</span>
              <span>{formatAmbulanceName(amb.id, index)}</span>
              <span className="camera-choice-power">{Math.round(amb.battery * 100)}%</span>
            </button>
          ))}
        </div>
      </section>

      {selectedDrone && (
        <div className="fpv-hud" aria-label={`${formatDroneName(selectedDrone.id)} flight telemetry`}>
          <span className="fpv-reticle" aria-hidden="true" />
          <div className="fpv-hud-top">
            <span>FPV // {formatDroneName(selectedDrone.id)}</span>
            <span>ALT {DRONE_FLIGHT_ALTITUDE}m</span>
          </div>
          <div className="fpv-hud-bottom">
            <span>SCANNER · ONLINE</span>
            <span>BAT {Math.round(selectedDrone.battery * 100)}%</span>
          </div>
        </div>
      )}

      {selectedAmbulance && (
        <div className="fpv-hud ambulance-hud" aria-label={`${formatAmbulanceName(selectedAmbulance.id)} telemetry`}>
          <span className="fpv-reticle ambulance-reticle" aria-hidden="true" />
          <div className="fpv-hud-top">
            <span>CHASE // {formatAmbulanceName(selectedAmbulance.id)}</span>
            <span>GND LEVEL</span>
          </div>
          <div className="fpv-hud-bottom">
            <span>{selectedAmbulance.carrying_victim ? '🚨 PATIENT ONBOARD' : 'DISPATCH · READY'}</span>
            <span>BAT {Math.round(selectedAmbulance.battery * 100)}%</span>
          </div>
        </div>
      )}

      <div className="tactical-legend" aria-label="Tactical map legend">
        <span><i className="legend-swatch legend-drone" /> Drone</span>
        <span><i className="legend-swatch legend-ambulance" /> Ambulance</span>
        <span><i className="legend-swatch legend-person" /> Victim</span>
        <span><i className="legend-swatch legend-hazard" /> Hazard</span>
      </div>
    </div>
  );
}

function TacticalScene({ worldState, agentTrails, selectedDroneId, selectedAmbulanceId }) {
  const sourceGrid = worldState.grid;
  const gridCache = useRef({ signature: '', grid: sourceGrid });
  const gridSignature = useMemo(
    () => sourceGrid.map((row) => row.join('')).join(''),
    [sourceGrid],
  );

  // State arrives at high frequency, while terrain changes only when a disaster
  // changes a cell. A stable reference avoids rebuilding 40k instance matrices.
  if (gridCache.current.signature !== gridSignature) {
    gridCache.current = { signature: gridSignature, grid: sourceGrid };
  }

  const grid = gridCache.current.grid;
  const gridHeight = grid.length;
  const gridWidth = grid[0].length;
  const centerRow = gridHeight / 2;
  const centerCol = gridWidth / 2;

  const terrain = useMemo(() => buildTerrain(grid, centerRow, centerCol), [grid, centerRow, centerCol]);
  const agents = useMemo(
    () => Object.entries(worldState.agents || {}).map(([id, agent]) => ({ id, ...agent })),
    [worldState.agents],
  );
  const victims = useMemo(
    () => Object.entries(worldState.victims || {}).map(([id, victim]) => ({ id, ...victim })),
    [worldState.victims],
  );
  const selectedDrone = agents.find((agent) => agent.id === selectedDroneId && agent.type === 'drone');
  const selectedAmbulance = agents.find((agent) => agent.id === selectedAmbulanceId && agent.type === 'ambulance');
  const hasPov = selectedDrone || selectedAmbulance;
  const worldSize = Math.max(gridWidth, gridHeight);

  return (
    <>
      <color attach="background" args={['#1d4d63']} />
      <fog attach="fog" args={['#1d4d63', worldSize * 0.58, worldSize * 1.1]} />
      <ambientLight intensity={1.22} />
      <hemisphereLight args={['#c5f0ff', '#426c50', 1.9]} />
      <directionalLight position={[45, 72, 32]} intensity={2.75} color="#fff2d8" />
      <directionalLight position={[-35, 30, -28]} intensity={0.95} color="#8edbff" />
      <Stars radius={worldSize * 2.2} depth={60} count={360} factor={1.4} saturation={0} fade speed={0.1} />

      <mesh position={[0, -0.13, 0]} receiveShadow>
        <boxGeometry args={[worldSize + 8, 0.18, worldSize + 8]} />
        <meshStandardMaterial color="#263f46" roughness={1} />
      </mesh>
      <gridHelper args={[worldSize, Math.ceil(worldSize / 10), '#3e91a8', '#285f6e']} position={[0, 0.01, 0]} />

      {Object.entries(terrain.cells).map(([type, cells]) => (
        cells.length > 0 && <TerrainInstancedMesh key={type} type={Number(type)} cells={cells} />
      ))}
      <WaterSurface cells={terrain.cells[5]} />
      <RoadMarkings marks={terrain.roadMarks} />
      <TreeLayer trees={terrain.trees} />
      <HazardLayer fires={terrain.fires} centerRow={centerRow} centerCol={centerCol} />

      {victims.filter((victim) => !victim.rescued).map((victim) => (
        <RescuePerson
          key={victim.id}
          victim={victim}
          cellType={grid[victim.row]?.[victim.col]}
          centerRow={centerRow}
          centerCol={centerCol}
        />
      ))}

      {/* FPV hides every drone mesh; ambulance POV hides the selected ambulance
          so the chase-cam doesn't clip through its own model. */}
      {agents.filter((agent) => {
        if (agent.type === 'traffic_light') return false;
        if (selectedDroneId && agent.type === 'drone') return false;
        if (selectedAmbulanceId && agent.id === selectedAmbulanceId) return false;
        return true;
      }).map((agent, index) => (
        <AgentModel
          key={agent.id}
          agent={agent}
          index={index}
          grid={grid}
          centerRow={centerRow}
          centerCol={centerCol}
          heading={headingFromTrail(agent.id, agentTrails)}
          isSelected={agent.id === selectedDroneId}
        />
      ))}

      {agents.filter((agent) => agent.type === 'drone').map((agent) => (
        <DroneSensorRing
          key={`${agent.id}-sensor`}
          agent={agent}
          centerRow={centerRow}
          centerCol={centerCol}
          active={agent.id === selectedDroneId}
        />
      ))}

      {selectedDrone ? (
        <DronePOVCamera
          agent={selectedDrone}
          heading={headingFromTrail(selectedDrone.id, agentTrails)}
          centerRow={centerRow}
          centerCol={centerCol}
          grid={grid}
        />
      ) : selectedAmbulance ? (
        <AmbulancePOVCamera
          agent={selectedAmbulance}
          heading={headingFromTrail(selectedAmbulance.id, agentTrails)}
          centerRow={centerRow}
          centerCol={centerCol}
          grid={grid}
        />
      ) : (
        <OverviewCamera worldSize={worldSize} />
      )}
      {!hasPov && <OrbitControls makeDefault enableDamping dampingFactor={0.09} maxPolarAngle={Math.PI / 2.05} minDistance={18} maxDistance={worldSize * 1.1} />}
    </>
  );
}

function buildTerrain(grid, centerRow, centerCol) {
  const cells = Object.fromEntries(Object.keys(TERRAIN_STYLE).map((type) => [type, []]));
  const roadMarks = [];
  const trees = [];
  const fires = [];

  for (let row = 0; row < grid.length; row += 1) {
    for (let col = 0; col < grid[row].length; col += 1) {
      const type = grid[row][col];
      if (cells[type]) cells[type].push({ row, col, x: col - centerCol, z: row - centerRow });

      if (type === 1 && hash2D(row, col, 9) > 0.63) {
        const horizontal = grid[row]?.[col - 1] === 1 || grid[row]?.[col + 1] === 1;
        const vertical = grid[row - 1]?.[col] === 1 || grid[row + 1]?.[col] === 1;
        if (horizontal !== vertical) roadMarks.push({ x: col - centerCol, z: row - centerRow, horizontal });
      }
      if (type === 9 && hash2D(row, col, 11) > 0.66) {
        trees.push({ x: col - centerCol, z: row - centerRow, scale: 0.55 + hash2D(row, col, 12) * 0.45 });
      }
      if (type === 6 && hash2D(row, col, 15) > 0.4) fires.push({ row, col });
    }
  }
  return { cells, roadMarks, trees, fires };
}

function TerrainInstancedMesh({ type, cells }) {
  const meshRef = useRef();
  const dummy = useMemo(() => new THREE.Object3D(), []);
  const color = useMemo(() => new THREE.Color(), []);
  const style = TERRAIN_STYLE[type];

  useEffect(() => {
    if (!meshRef.current) return;
    cells.forEach((cell, index) => {
      const height = terrainHeight(type, cell.row, cell.col);
      dummy.position.set(cell.x, height / 2, cell.z);
      dummy.scale.set(0.97, height, 0.97);
      dummy.updateMatrix();
      meshRef.current.setMatrixAt(index, dummy.matrix);

      const variation = 0.84 + hash2D(cell.row, cell.col, type) * 0.25;
      color.set(style.color).multiplyScalar(variation);
      meshRef.current.setColorAt(index, color);
    });
    meshRef.current.instanceMatrix.needsUpdate = true;
    if (meshRef.current.instanceColor) meshRef.current.instanceColor.needsUpdate = true;
  }, [cells, color, dummy, style.color, type]);

  return (
    <instancedMesh ref={meshRef} args={[null, null, cells.length]} frustumCulled={false}>
      <boxGeometry args={[1, 1, 1]} />
      <meshStandardMaterial
        vertexColors
        roughness={style.roughness}
        metalness={style.metalness || 0}
        emissive={type === 5 ? '#08789f' : type === 8 ? '#07584f' : type === 2 ? '#17435d' : '#000000'}
        emissiveIntensity={type === 5 ? 0.5 : type === 8 ? 0.35 : type === 2 ? 0.32 : 0}
        transparent={type === 5}
        opacity={type === 5 ? 0.92 : 1}
      />
    </instancedMesh>
  );
}

function WaterSurface({ cells = [] }) {
  const meshRef = useRef();
  const dummy = useMemo(() => new THREE.Object3D(), []);

  useEffect(() => {
    if (!meshRef.current) return;
    cells.forEach((cell, index) => {
      dummy.position.set(cell.x, 0.225, cell.z);
      dummy.rotation.set(-Math.PI / 2, 0, 0);
      dummy.scale.set(0.985, 0.985, 1);
      dummy.updateMatrix();
      meshRef.current.setMatrixAt(index, dummy.matrix);
    });
    meshRef.current.instanceMatrix.needsUpdate = true;
  }, [cells, dummy]);

  if (!cells.length) return null;
  return (
    <instancedMesh ref={meshRef} args={[null, null, cells.length]} renderOrder={1} frustumCulled={false}>
      <planeGeometry args={[1, 1]} />
      <meshPhysicalMaterial
        color="#0c9ed2"
        emissive="#07517d"
        emissiveIntensity={0.34}
        metalness={0.16}
        roughness={0.28}
        clearcoat={0.3}
        clearcoatRoughness={0.28}
        transparent
        opacity={0.94}
      />
    </instancedMesh>
  );
}

function RoadMarkings({ marks }) {
  const meshRef = useRef();
  const dummy = useMemo(() => new THREE.Object3D(), []);

  useEffect(() => {
    if (!meshRef.current) return;
    marks.forEach((mark, index) => {
      dummy.position.set(mark.x, 0.165, mark.z);
      dummy.scale.set(mark.horizontal ? 0.35 : 0.07, 1, mark.horizontal ? 0.07 : 0.35);
      dummy.updateMatrix();
      meshRef.current.setMatrixAt(index, dummy.matrix);
    });
    meshRef.current.instanceMatrix.needsUpdate = true;
  }, [dummy, marks]);

  if (!marks.length) return null;
  return (
    <instancedMesh ref={meshRef} args={[null, null, marks.length]} frustumCulled={false}>
      <boxGeometry args={[1, 0.015, 1]} />
      <meshStandardMaterial color="#f5c85b" emissive="#573b0a" emissiveIntensity={0.35} roughness={0.55} />
    </instancedMesh>
  );
}

function TreeLayer({ trees }) {
  const trunkRef = useRef();
  const crownRef = useRef();
  const dummy = useMemo(() => new THREE.Object3D(), []);

  useEffect(() => {
    if (!trunkRef.current || !crownRef.current) return;
    trees.forEach((tree, index) => {
      dummy.position.set(tree.x, 0.34 * tree.scale, tree.z);
      dummy.scale.set(tree.scale, tree.scale, tree.scale);
      dummy.updateMatrix();
      trunkRef.current.setMatrixAt(index, dummy.matrix);

      dummy.position.set(tree.x, 0.87 * tree.scale, tree.z);
      dummy.scale.set(tree.scale, tree.scale, tree.scale);
      dummy.updateMatrix();
      crownRef.current.setMatrixAt(index, dummy.matrix);
    });
    trunkRef.current.instanceMatrix.needsUpdate = true;
    crownRef.current.instanceMatrix.needsUpdate = true;
  }, [dummy, trees]);

  if (!trees.length) return null;
  return (
    <>
      <instancedMesh ref={trunkRef} args={[null, null, trees.length]} frustumCulled={false}>
        <cylinderGeometry args={[0.07, 0.09, 0.68, 6]} />
        <meshStandardMaterial color="#5e4031" roughness={1} />
      </instancedMesh>
      <instancedMesh ref={crownRef} args={[null, null, trees.length]} frustumCulled={false}>
        <coneGeometry args={[0.42, 0.9, 7]} />
        <meshStandardMaterial color="#5d9b66" roughness={0.9} />
      </instancedMesh>
    </>
  );
}

function HazardLayer({ fires, centerRow, centerCol }) {
  return fires.map((fire) => {
    const [x, z] = cellToWorld(fire.row, fire.col, centerRow, centerCol);
    const scale = 0.45 + hash2D(fire.row, fire.col, 22) * 0.3;
    return (
      <group key={`${fire.row}-${fire.col}`} position={[x, 0.4, z]} scale={scale}>
        <mesh>
          <coneGeometry args={[0.48, 1.5, 6]} />
          <meshStandardMaterial color="#ff6422" emissive="#ff2500" emissiveIntensity={2.5} roughness={0.7} />
        </mesh>
        <mesh position={[0.1, 0.28, 0]} scale={0.5}>
          <coneGeometry args={[0.42, 1.35, 5]} />
          <meshBasicMaterial color="#ffd166" transparent opacity={0.92} />
        </mesh>
      </group>
    );
  });
}

function RescuePerson({ victim, cellType, centerRow, centerCol }) {
  const [x, z] = cellToWorld(victim.row, victim.col, centerRow, centerCol);
  const ground = terrainHeight(cellType ?? 0, victim.row, victim.col);
  const color = VICTIM_COLORS[victim.severity_name] || '#f3cfb2';
  const rotation = hash2D(victim.row, victim.col, 33) * Math.PI * 2;

  return (
    <group position={[x, ground, z]} rotation={[0, rotation, 0]}>
      {victim.detected && (
        <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.025, 0]}>
          <ringGeometry args={[0.36, 0.41, 20]} />
          <meshBasicMaterial color={color} transparent opacity={0.94} side={THREE.DoubleSide} />
        </mesh>
      )}
      <mesh position={[0, 0.56, 0]}>
        <capsuleGeometry args={[0.16, 0.34, 4, 8]} />
        <meshStandardMaterial color={color} roughness={0.72} />
      </mesh>
      <mesh position={[0, 1.02, 0]}>
        <sphereGeometry args={[0.17, 10, 10]} />
        <meshStandardMaterial color="#f2c5a0" roughness={0.9} />
      </mesh>
      <mesh position={[-0.2, 0.65, 0]} rotation={[0, 0, -0.65]}>
        <cylinderGeometry args={[0.035, 0.035, 0.38, 6]} />
        <meshStandardMaterial color="#f2c5a0" />
      </mesh>
      <mesh position={[0.2, 0.65, 0]} rotation={[0, 0, 0.65]}>
        <cylinderGeometry args={[0.035, 0.035, 0.38, 6]} />
        <meshStandardMaterial color="#f2c5a0" />
      </mesh>
    </group>
  );
}

function AgentModel({ agent, index, grid, centerRow, centerCol, heading, isSelected }) {
  const [x, z] = cellToWorld(agent.row, agent.col, centerRow, centerCol);
  const roundedRow = Math.round(agent.row);
  const roundedCol = Math.round(agent.col);
  const terrain = grid[roundedRow]?.[roundedCol] ?? 0;

  if (agent.type === 'drone') {
    const ground = terrainHeight(terrain, roundedRow, roundedCol);
    return <DroneModel position={[x, ground + DRONE_FLIGHT_ALTITUDE, z]} heading={heading} selected={isSelected} />;
  }
  // Ground vehicles use capped height to stay at street level.
  const ground = groundVehicleHeight(terrain, roundedRow, roundedCol);
  if (agent.type === 'ambulance') {
    return <AmbulanceModel position={[x, ground + 0.48, z]} heading={heading} carrying={Boolean(agent.carrying_victim)} />;
  }
  return <GroundRobotModel position={[x, ground + 0.32, z]} heading={heading} index={index} />;
}

function DroneModel({ position, heading, selected }) {
  const propellerRef = useRef();

  useFrame((_, delta) => {
    if (propellerRef.current) propellerRef.current.rotation.y += delta * 26;
  });

  return (
    <group position={position} rotation={[0, heading, 0]}>
      {selected && <pointLight color="#48e8ff" intensity={5} distance={12} decay={2} />}
      {/* Crossed carbon-fiber arms make the aerial unit read as a quadcopter. */}
      <mesh rotation={[0, Math.PI / 4, 0]}>
        <boxGeometry args={[1.78, 0.075, 0.09]} />
        <meshStandardMaterial color="#162a35" metalness={0.85} roughness={0.22} />
      </mesh>
      <mesh rotation={[0, -Math.PI / 4, 0]}>
        <boxGeometry args={[1.78, 0.075, 0.09]} />
        <meshStandardMaterial color="#162a35" metalness={0.85} roughness={0.22} />
      </mesh>
      <mesh scale={[1.08, 0.38, 1.36]}>
        <sphereGeometry args={[0.31, 14, 10]} />
        <meshStandardMaterial color={selected ? '#8debf6' : '#2d9eb9'} metalness={0.76} roughness={0.24} emissive={selected ? '#0d7184' : '#073444'} emissiveIntensity={0.9} />
      </mesh>
      <mesh position={[0, 0.14, -0.03]} scale={[0.9, 0.34, 1.12]}>
        <sphereGeometry args={[0.23, 12, 8]} />
        <meshStandardMaterial color="#162b38" metalness={0.82} roughness={0.18} emissive="#0a3648" emissiveIntensity={0.5} />
      </mesh>
      <mesh position={[0, -0.19, 0.36]}>
        <sphereGeometry args={[0.15, 10, 8]} />
        <meshStandardMaterial color="#0c1d27" metalness={0.8} roughness={0.15} emissive="#126c83" emissiveIntensity={0.55} />
      </mesh>
      <group ref={propellerRef}>
        {[[0.62, 0.62], [-0.62, 0.62], [0.62, -0.62], [-0.62, -0.62]].map(([px, pz]) => (
          <group key={`${px}-${pz}`} position={[px, 0.09, pz]}>
            <mesh position={[0, -0.1, 0]}>
              <cylinderGeometry args={[0.075, 0.075, 0.16, 10]} />
              <meshStandardMaterial color="#0f222c" metalness={0.82} roughness={0.18} />
            </mesh>
            <mesh rotation={[0, 0.4, 0]}>
              <boxGeometry args={[0.58, 0.018, 0.055]} />
              <meshStandardMaterial color="#c7f8ff" transparent opacity={0.68} metalness={0.35} />
            </mesh>
          </group>
        ))}
      </group>
      {[-0.23, 0.23].map((x) => (
        <mesh key={x} position={[x, -0.29, -0.06]} rotation={[0, 0, x > 0 ? -0.38 : 0.38]}>
          <cylinderGeometry args={[0.026, 0.026, 0.52, 6]} />
          <meshStandardMaterial color="#172b36" metalness={0.65} />
        </mesh>
      ))}
      <mesh position={[0, -0.42, -0.06]} rotation={[0, 0, Math.PI / 2]}>
        <cylinderGeometry args={[0.022, 0.022, 0.58, 6]} />
        <meshStandardMaterial color="#172b36" metalness={0.65} />
      </mesh>
    </group>
  );
}

function AmbulanceModel({ position, heading, carrying }) {
  return (
    <group position={position} rotation={[0, heading, 0]}>
      {carrying && <pointLight color="#ff465a" intensity={2.8} distance={5} />}
      <mesh>
        <boxGeometry args={[0.78, 0.48, 1.32]} />
        <meshStandardMaterial color="#edf4f7" metalness={0.22} roughness={0.5} />
      </mesh>
      <mesh position={[0, 0.05, -0.26]}>
        <boxGeometry args={[0.82, 0.1, 0.72]} />
        <meshStandardMaterial color="#df4251" metalness={0.16} roughness={0.45} />
      </mesh>
      <mesh position={[0, 0.3, -0.16]}>
        <boxGeometry args={[0.15, 0.04, 0.42]} />
        <meshBasicMaterial color="#ef4454" />
      </mesh>
      {[-0.42, 0.42].map((wheelZ) => [-0.44, 0.44].map((wheelX) => (
        <mesh key={`${wheelX}-${wheelZ}`} position={[wheelX, -0.21, wheelZ]} rotation={[0, Math.PI / 2, 0]}>
          <cylinderGeometry args={[0.17, 0.17, 0.1, 10]} />
          <meshStandardMaterial color="#10171c" roughness={0.88} />
        </mesh>
      )))}
    </group>
  );
}

function GroundRobotModel({ position, heading, index }) {
  return (
    <group position={position} rotation={[0, heading, 0]}>
      <mesh>
        <boxGeometry args={[0.68, 0.36, 0.68]} />
        <meshStandardMaterial color="#35bd8c" metalness={0.45} roughness={0.38} />
      </mesh>
      <mesh position={[0, 0.26, -0.12]}>
        <sphereGeometry args={[0.19, 10, 8]} />
        <meshStandardMaterial color="#183c42" metalness={0.7} roughness={0.2} emissive="#0a7b87" emissiveIntensity={0.6} />
      </mesh>
      <mesh position={[0, 0.02, -0.36]}>
        <boxGeometry args={[0.3, 0.09, 0.08]} />
        <meshBasicMaterial color={index % 2 ? '#aaf7ff' : '#f8db75'} />
      </mesh>
    </group>
  );
}

function DroneSensorRing({ agent, centerRow, centerCol, active }) {
  const [x, z] = cellToWorld(agent.row, agent.col, centerRow, centerCol);
  return (
    <mesh position={[x, 0.17, z]} rotation={[-Math.PI / 2, 0, 0]}>
      <ringGeometry args={[7.78, 8, 48]} />
      <meshBasicMaterial color={active ? '#7eeeff' : '#3b9eb2'} transparent opacity={active ? 0.62 : 0.2} side={THREE.DoubleSide} />
    </mesh>
  );
}

function DronePOVCamera({ agent, heading, centerRow, centerCol, grid }) {
  const { camera } = useThree();
  const currentPos = useRef(new THREE.Vector3());
  const currentHeading = useRef(heading);
  const firstFrame = useRef(true);

  useEffect(() => {
    camera.fov = 67;
    camera.updateProjectionMatrix();
    firstFrame.current = true;
  }, [agent.id, camera]);

  useFrame((_, delta) => {
    const [targetX, targetZ] = cellToWorld(agent.row, agent.col, centerRow, centerCol);
    const targetRow = Math.round(agent.row);
    const targetCol = Math.round(agent.col);
    const localTerrain = grid[targetRow]?.[targetCol] ?? 0;
    const targetGround = terrainHeight(localTerrain, targetRow, targetCol);

    let headingDiff = heading - currentHeading.current;
    while (headingDiff > Math.PI) headingDiff -= Math.PI * 2;
    while (headingDiff < -Math.PI) headingDiff += Math.PI * 2;

    if (firstFrame.current) {
      currentPos.current.set(targetX, targetGround, targetZ);
      currentHeading.current = heading;
    } else {
      const posLerp = 1 - Math.exp(-delta * 10);
      currentPos.current.x += (targetX - currentPos.current.x) * posLerp;
      currentPos.current.y += (targetGround - currentPos.current.y) * posLerp;
      currentPos.current.z += (targetZ - currentPos.current.z) * posLerp;
      currentHeading.current += headingDiff * (1 - Math.exp(-delta * 8));
    }

    const x = currentPos.current.x;
    const y = currentPos.current.y;
    const z = currentPos.current.z;
    const h = currentHeading.current;

    const targetPosition = new THREE.Vector3(x, y + DRONE_FLIGHT_ALTITUDE + 0.7, z);
    
    if (firstFrame.current) {
      camera.position.copy(targetPosition);
      firstFrame.current = false;
    } else {
      camera.position.lerp(targetPosition, 1 - Math.exp(-delta * 15));
    }
    
    const forward = new THREE.Vector3(Math.sin(h), -0.55, Math.cos(h)).normalize();
    camera.lookAt(camera.position.clone().add(forward.multiplyScalar(26)));
  });

  return null;
}

function AmbulancePOVCamera({ agent, heading, centerRow, centerCol, grid }) {
  const { camera } = useThree();
  const currentPos = useRef(new THREE.Vector3());
  const currentHeading = useRef(heading);
  const firstFrame = useRef(true);

  useEffect(() => {
    camera.fov = 62;
    camera.updateProjectionMatrix();
    firstFrame.current = true;
  }, [agent.id, camera]);

  useFrame((_, delta) => {
    const [targetX, targetZ] = cellToWorld(agent.row, agent.col, centerRow, centerCol);
    const targetRow = Math.round(agent.row);
    const targetCol = Math.round(agent.col);
    const localTerrain = grid[targetRow]?.[targetCol] ?? 0;
    const targetGround = groundVehicleHeight(localTerrain, targetRow, targetCol);
    
    // Calculate the shortest angular distance for heading interpolation
    let headingDiff = heading - currentHeading.current;
    while (headingDiff > Math.PI) headingDiff -= Math.PI * 2;
    while (headingDiff < -Math.PI) headingDiff += Math.PI * 2;
    
    if (firstFrame.current) {
      currentPos.current.set(targetX, targetGround, targetZ);
      currentHeading.current = heading;
    } else {
      // Lerp the ambulance position exactly like GroundRobotModel does
      const posLerp = 1 - Math.exp(-delta * 10);
      currentPos.current.x += (targetX - currentPos.current.x) * posLerp;
      currentPos.current.y += (targetGround - currentPos.current.y) * posLerp;
      currentPos.current.z += (targetZ - currentPos.current.z) * posLerp;
      currentHeading.current += headingDiff * (1 - Math.exp(-delta * 8));
    }
    
    const x = currentPos.current.x;
    const y = currentPos.current.y;
    const z = currentPos.current.z;
    const h = currentHeading.current;

    // Find a clear camera position by shrinking the behind distance if it clips a building
    let targetDistance = AMBULANCE_CAMERA_BEHIND;
    let behindX, behindZ, camRow, camCol, camTerrain;
    
    // Step inward until we find a clear cell (not a building)
    for (let step = 0; step < 8; step++) {
      behindX = x - Math.sin(h) * targetDistance;
      behindZ = z - Math.cos(h) * targetDistance;
      
      camRow = Math.round(behindZ + centerRow);
      camCol = Math.round(behindX + centerCol);
      camTerrain = grid[camRow]?.[camCol] ?? 0;
      
      if (camTerrain !== 2 && camTerrain !== 3) {
        break; // Clear!
      }
      targetDistance -= 1.0;
    }

    const targetPosition = new THREE.Vector3(behindX, y + AMBULANCE_CAMERA_HEIGHT, behindZ);

    if (firstFrame.current) {
      camera.position.copy(targetPosition);
      firstFrame.current = false;
    } else {
      // Camera tightly follows the smoothed targetPosition
      camera.position.lerp(targetPosition, 1 - Math.exp(-delta * 15));
    }

    // Look at a point ahead of the smoothed ambulance
    const aheadX = x + Math.sin(h) * 8;
    const aheadZ = z + Math.cos(h) * 8;
    const lookTarget = new THREE.Vector3(aheadX, y + 0.9, aheadZ);
    camera.lookAt(lookTarget);
  });

  return null;
}

function OverviewCamera({ worldSize }) {
  const { camera } = useThree();

  useEffect(() => {
    camera.fov = 46;
    camera.position.set(0, worldSize * 0.48, worldSize * 0.34);
    camera.lookAt(0, 0, 0);
    camera.updateProjectionMatrix();
  }, [camera, worldSize]);

  return null;
}

function formatDroneName(id, fallbackIndex = 0) {
  const match = id.match(/(\d+)$/);
  return `DRONE-${String(match ? Number(match[1]) + 1 : fallbackIndex + 1).padStart(2, '0')}`;
}

function formatAmbulanceName(id, fallbackIndex = 0) {
  const match = id.match(/(\d+)$/);
  return `AMBU-${String(match ? Number(match[1]) + 1 : fallbackIndex + 1).padStart(2, '0')}`;
}
