import React, { useMemo, useRef, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';

// 0: EMPTY, 1: ROAD, 2: BUILDING, 3: PARK, 4: WATER
// 5: HOSPITAL, 6: CHARGING_STATION, 7: DEBRIS
// 8: BUILDING_DAMAGED, 9: FIRE

const TERRAIN_COLORS = {
  0: '#222222', // Empty
  1: '#333333', // Road
  2: '#555555', // Building
  3: '#8a2b2b', // Damaged Building
  4: '#8b5a2b', // Debris
  5: '#1a5b7c', // Water
  6: '#ff4500', // Fire
  7: '#ffffff', // Hospital
  8: '#ffff00', // Charging Station
  9: '#2d4c2b'  // Park
};

const TERRAIN_HEIGHTS = {
  0: 0.05, 1: 0.05, 2: 1.0, 3: 0.5, 4: 0.05,
  5: 0.1, 6: 1.0, 7: 1.5, 8: 0.05, 9: 0.05
};

export default function TacticalView({ worldState }) {
  if (!worldState || !worldState.grid) return <div style={{color:'white'}}>Loading...</div>;

  const gridHeight = worldState.grid.length;
  const gridWidth = worldState.grid[0].length;
  const centerR = gridHeight / 2;
  const centerC = gridWidth / 2;

  // Pre-calculate instances for each terrain type
  const instances = useMemo(() => {
    const map = {};
    Object.keys(TERRAIN_COLORS).forEach(k => map[k] = []);
    
    worldState.grid.forEach((row, r) => {
      row.forEach((val, c) => {
        if (map[val] !== undefined) {
          map[val].push({ r: r - centerR, c: c - centerC });
        }
      });
    });
    return map;
  }, [worldState.grid]);

  return (
    <div style={{ width: '100%', height: '600px', background: '#000', borderRadius: '8px', overflow: 'hidden' }}>
      <Canvas camera={{ position: [0, 30, 30], fov: 45 }}>
        <ambientLight intensity={0.5} />
        <directionalLight position={[10, 20, 10]} intensity={1} castShadow />
        
        {/* Render Instanced Grids */}
        {Object.entries(instances).map(([type, cells]) => (
           cells.length > 0 && <TerrainInstancedMesh key={type} type={type} cells={cells} />
        ))}

        {/* Render Agents (Regular meshes because there are few and they move constantly) */}
        {worldState.agents && Object.values(worldState.agents).map(agent => (
          <mesh 
            key={agent.id} 
            position={[agent.col - centerC, 0.5, agent.row - centerR]}
          >
            <sphereGeometry args={[0.4, 16, 16]} />
            <meshStandardMaterial color={agent.type === 'ambulance' ? 'red' : 'cyan'} />
          </mesh>
        ))}

        <OrbitControls />
      </Canvas>
    </div>
  );
}

function TerrainInstancedMesh({ type, cells }) {
  const meshRef = useRef();
  const dummy = useMemo(() => new THREE.Object3D(), []);

  useEffect(() => {
    if (meshRef.current) {
      cells.forEach((cell, i) => {
        dummy.position.set(cell.c, TERRAIN_HEIGHTS[type] / 2, cell.r);
        dummy.updateMatrix();
        meshRef.current.setMatrixAt(i, dummy.matrix);
      });
      meshRef.current.instanceMatrix.needsUpdate = true;
    }
  }, [cells, type, dummy]);

  return (
    <instancedMesh ref={meshRef} args={[null, null, cells.length]}>
      <boxGeometry args={[0.95, TERRAIN_HEIGHTS[type], 0.95]} />
      <meshStandardMaterial 
        color={TERRAIN_COLORS[type]} 
        emissive={type === '9' ? TERRAIN_COLORS[type] : '#000000'}
        emissiveIntensity={type === '9' ? 2 : 0}
        transparent={type === '4'}
        opacity={type === '4' ? 0.8 : 1.0}
      />
    </instancedMesh>
  );
}
