# Deep Dive Analysis & Fixes

I performed a thorough investigation into the physics engine, the ambulance logic, the dashboard display, and the world generation, and discovered the root causes of the issues you experienced. Here is a breakdown of what was actually going wrong and the fixes I applied.

## 1. Map Randomization
**The Bug:** The map, hospital placement, and victim spawn locations were identical every time you restarted the server, which meant drones repeatedly spawned in exactly the same tricky corners, repeating their behavior.
**The Cause:** The `SimRunner` had a hardcoded `seed = 42` for the `Config`.
**The Fix:** The server now passes a completely randomized seed between 0 and 999999 to the generator, meaning you'll get a uniquely diverse city layout and victim spread on every run!

## 2. The Invisible Walls (Ambulances Getting Stuck)
**The Bug:** Ambulances would successfully rescue a victim and calculate a perfectly valid A* path to the hospital. However, as soon as they tried to move through an intersection, they would get permanently stuck. 
**The Cause:** Traffic lights act as stationary agents located at road intersections. Our `physics.py` collision detection system treated traffic lights as "ground units" and prevented any other ground unit (like an ambulance) from entering their space. Because A* only checked terrain passability and didn't know about the traffic light agents blocking the intersections, it would route ambulances right into them. When the physics engine rejected the move, the ambulance would get stuck forever trying to proceed.
**The Fix:** I updated `physics.py` to explicitly exclude `traffic_light` agents from collision detection so ambulances can pass through intersections freely.

## 3. Power Displaying 100%
**The Bug:** You noticed drones were running out of power too early in previous runs, but the dashboard recently showed "Fleet Power 100%".
**The Cause:** `avg_battery` was calculating the average battery across **all active agents**. Since the city spawns hundreds of stationary traffic lights with infinite battery (100%), they completely skewed the average, meaning the dashboard almost always showed 100% power even when drones were dying. Also, in the `physics.py` engine, I found that the battery was properly draining by `battery_drain_rate` every tick.
**The Fix:** I updated the calculation in `world_state.py` to only average the battery levels of the actual active swarm fleet (`drone`, `ambulance`, `boat`), ignoring stationary infrastructure. The dashboard will now accurately reflect the power levels of the drones!

With these edge cases eliminated, you should now see a diverse, randomly generated map with fully operational ambulances traversing intersections smoothly to the hospital!
