# SatQuery AI: Verification Dump & Forensic Evidence Audit
**Generated:** 2026-10-14T20:15:52.935682  
**Target Report File:** `reports/verification_dump.md`  
**Audit Protocol:** Raw execution evidence, exact outputs, and unvarnished truth collection.

---

## 1. REPO STATE

### 1.1 Git Status
```
On branch main
Your branch is up to date with 'origin/main'.

Changes to be committed:
  (use "git restore --staged <file>..." to unstage)
	renamed:    DEPLOYMENT_GUIDE.md -> docs/DEPLOYMENT_GUIDE.md
	renamed:    FINAL_DEMO_QUERIES.md -> docs/FINAL_DEMO_QUERIES.md
	renamed:    FINAL_POSTER_CORRECTIONS.md -> docs/FINAL_POSTER_CORRECTIONS.md
	renamed:    FINAL_PROJECT_AUDIT.md -> docs/FINAL_PROJECT_AUDIT.md
	renamed:    MODEL_AUDIT_FACTS.json -> docs/MODEL_AUDIT_FACTS.json
	renamed:    MODEL_AUDIT_REPORT.md -> docs/MODEL_AUDIT_REPORT.md
	renamed:    PROJECT_CONTEXT.md -> docs/PROJECT_CONTEXT.md
	renamed:    SIH_2026_PRESENTATION_MASTER_DECK.txt -> docs/SIH_2026_PRESENTATION_MASTER_DECK.txt
	renamed:    VIDEO_PITCH_SCRIPT.md -> docs/VIDEO_PITCH_SCRIPT.md
	deleted:    ml_pipeline/generate_dummy_data.py

Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
	modified:   README.md
	modified:   backend/agent_controller.py
	modified:   backend/data_models.py
	modified:   backend/main.py
	modified:   backend/ml_models.py
	modified:   backend/model_interfaces.py
	modified:   backend/object_detection.py
	modified:   backend/query_understanding.py
	modified:   backend/storage_manager.py
	modified:   backend/test_query_routing.py
	modified:   docs/FINAL_POSTER_CORRECTIONS.md
	modified:   docs/FINAL_PROJECT_AUDIT.md
	modified:   docs/MODEL_AUDIT_FACTS.json
	modified:   docs/MODEL_AUDIT_REPORT.md
	modified:   docs/SIH_2026_PRESENTATION_MASTER_DECK.txt
	modified:   docs/VIDEO_PITCH_SCRIPT.md
	modified:   frontend/next.config.ts
	modified:   frontend/public/demo/risat1_cband_radar_sar.png
	modified:   frontend/src/app/dashboard/page.tsx
	modified:   frontend/src/app/page.tsx
	modified:   frontend/src/app/query/page.tsx
	modified:   ml_pipeline/datasets.py
	modified:   ml_pipeline/train_vqa.py
	modified:   run_all.ps1

Untracked files:
  (use "git add <file>..." to include in what will be committed)
	.github/workflows/ci.yml
	antigravity_prompt_satquery.md
	backend/test_change_pipeline.py
	backend/test_cross_modal.py
	backend/test_input_robustness.py
	backend/test_tool_registry.py
	backend/tiling.py
	backend/tool_registry.py
	docs/DEMO_SCRIPT_230_PRECISE.md
	docs/DEMO_SCRIPT_2MIN30SEC.md
	docs/SIH_2026_PRESENTATION_VERIFIED_DECK.md
	docs/model_choice.md
	ml_pipeline/README.md
	ml_pipeline/change_model.py
	ml_pipeline/cross_modal_model.py
	ml_pipeline/data/
	ml_pipeline/data_sanity.py
	ml_pipeline/dataset_cache/
	ml_pipeline/eval_change.py
	ml_pipeline/eval_cross_modal.py
	ml_pipeline/eval_isro.py
	ml_pipeline/eval_routing.py
	ml_pipeline/eval_single_image.py
	ml_pipeline/profile_latency.py
	ml_pipeline/train_change_detector.py
	ml_pipeline/train_cross_modal.py
	ml_pipeline/train_rs_vlm.py
	reports/
	scratch/
	scripts/
	tests/
```

### 1.2 Git Log (Last 30 Commits)
```
98e155a feat(presets): update demo queries to exactly match ISRO representative queries from PS 26167
efad6ec docs: add simplified benchmark explanation to video script
abee772 docs: add official 5-minute SIH video pitch and demonstration script
ea94cf3 feat(backend): elevate validation and limitation messaging to production-grade remote sensing terminology
ee20856 perf(frontend): disable heavy NASA GIBS WMS tile layer over MapExplorer to resolve extreme load times and timeouts
0e79bb6 docs: create official README detailing fulfillment of ISRO/SAC PS 26167 evaluation criteria
b096885 fix(frontend): hide input compatibility badge for single-image queries where spatial correlation is not applicable
ed54179 fix(frontend): correctly fetch demo images from URLs instead of failing to parse as base64 data
9d43840 style: change YOUR DEMO to TRY DEMO on homepage
e09a438 style: change ENTER SYSTEM to YOUR DEMO on homepage
50120c5 style(launch): apply clean monochrome aerospace HUD design, removing excessive accent colors
fb506c3 feat(launch): implement perpetual orbit drift at 100% scroll with shortest-path reverse snapping
60a87bd perf(launch): optimize layout thrashing and bidirectional scroll smoothness
4372079 feat(gallery): remove 3D satellite globe and optimize rocket launch sequence
9655b08 feat(ui): restore Satellite Map and Sign In buttons in navbar and hero CTA
7475514 style(globe): reduce globe display size and slow auto-rotation speed for balanced hero framing
56aecf7 fix(globe): enforce strict 1:1 circular aspect ratio preventing vertical egg/oval distortion
6125ee0 feat(ui): restore minimal hero layout with pristine cinematic rotating Earth and ENTER SYSTEM button
35aa4c2 fix: restore 3D Earth globe, optimize local homepage images, and fix gallery Suspense crash
da3992c feat(map): add freehand lasso drawing tool for area selection
dd54ad5 fix(pwa): fix notch and status bar collision with header in standalone PWA mode
9cd9927 fix(map): remove popup obstruction on zoom and zoom out to normal overview on search cancel
5c4753c feat(map): enable polygon, rectangle, and circle area selection on satellite map
c6bc66a feat(map): remove search pin on clear and support arbitrary & random degrees rotation
04e55c2 fix(ui): remove telemetry badge from hero and mission cards from query workspace
dbfc992 feat(map): add 360-degree map rotation with aerospace compass HUD and auto-zoom place search box
cc20511 fix(mobile,pwa): optimize layouts, eliminate black voids, add cosmic depth, and enhance viewport ergonomics
c84495e feat(pwa): trigger 1-click direct native install dialog without intermediary popups
5e47847 fix(ui): remove floating install popup banner and PWA badge tags
302c0ba fix(pwa): ensure install button is always visible on phone browsers and restyle to match design specification
```

### 1.3 Git Branches
```
* main
  remotes/origin/main
```

### 1.4 Git Remotes
```
origin	https://github.com/atharva081106/SatQuery-AI.git (fetch)
origin	https://github.com/atharva081106/SatQuery-AI.git (push)
```

### 1.5 Push Status to Origin
- **HEAD Commit:** `98e155a875fb987d58b74665809fc2d1cfd7b87a`
- **origin/main Commit:** `98e155a875fb987d58b74665809fc2d1cfd7b87a`
- **Pushed to Origin:** YES (HEAD matches origin/main commit, but local working tree has uncommitted staged and unstaged modifications)
- **Uncommitted Modifications:** Staged renames in `docs/` and deleted `ml_pipeline/generate_dummy_data.py`; unstaged modifications in `README.md`, `backend/`, `frontend/`, `reports/`, and untracked `reports/` and `tests/` artifacts.

### 1.6 Full File Tree (Max Depth 4, Excluding node_modules, .git, .next, venv)
```
.
|-- .agents/
|   |-- rules/
|   |   \-- graphify.md
|   \-- workflows/
|       \-- graphify.md
|-- .github/
|   \-- workflows/
|       |-- ci.yml
|       \-- keep_alive.yml
|-- .gitignore
|-- .huggingface_cache/
|   |-- .agent_harnesses.json
|   |-- hub/
|   |   |-- .locks/
|   |   |   |-- models--Salesforce--blip-image-captioning-base/
|   |   |   |-- models--Salesforce--blip-vqa-base/
|   |   |   \-- models--google--owlvit-base-patch32/
|   |   |-- CACHEDIR.TAG
|   |   |-- models--Salesforce--blip-image-captioning-base/
|   |   |   |-- .no_exist/
|   |   |   |-- blobs/
|   |   |   |-- refs/
|   |   |   \-- snapshots/
|   |   |-- models--Salesforce--blip-vqa-base/
|   |   |   |-- .no_exist/
|   |   |   |-- blobs/
|   |   |   |-- refs/
|   |   |   \-- snapshots/
|   |   \-- models--google--owlvit-base-patch32/
|   |       |-- .no_exist/
|   |       |-- blobs/
|   |       |-- refs/
|   |       \-- snapshots/
|   \-- xet/
|       |-- https___cas_serv-tGqkUaZf_CBPHQ6h/
|       |   \-- staging/
|       \-- logs/
|           |-- xet_20260901T234910870+0530_7232.log
|           |-- xet_20260902T102713633+0530_4684.log
|           |-- xet_20260903T041903109+0530_32840.log
|           |-- xet_20260903T043232129+0530_29368.log
|           |-- xet_20260903T043325109+0530_9304.log
|           |-- xet_20260903T043505652+0530_15000.log
|           \-- xet_20260903T044225508+0530_16136.log
|-- README.md
|-- antigravity_prompt_satquery.md
|-- backend/
|   |-- .env
|   |-- Dockerfile
|   |-- Procfile
|   |-- __init__.py
|   |-- agent_controller.py
|   |-- analysis_planner.py
|   |-- answer_generator.py
|   |-- cache/
|   |   |-- 020ccebe2faf5d9817d299547818dae3.json
|   |   |-- 074189f2669bde5030239ee5d3c38096.json
|   |   |-- 0908dd84ab6700ed890624b4249f4dff.json
|   |   |-- 0d4897e6f7cd00a7ba43635f25430b38.json
|   |   |-- 105b92d9ac753afc99febdafea35b2b5.json
|   |   |-- 106875fffd1b803d01e2259f8356cec2.json
|   |   |-- 11bb28b5b3358c9db6c4f61938766b16.json
|   |   |-- 123e4da07a87c1b3c662527592403ba3.json
|   |   |-- 129495267b9df7e507db5d59ca71af3c.json
|   |   |-- 13b2dc52be5d5bcea94ada35206987df.json
|   |   |-- 14d1a24095c5960e32d9b353e6bf195c.json
|   |   |-- 14e01a1f016fd7a2983c678be2b4997e.json
|   |   |-- 14e058f9146a3890d57b5e81c4511104.json
|   |   |-- 156eb327e379e676c7d623bdcae5f2f4.json
|   |   |-- 19e1dc7076f4cf12615a44fe0e72cbfb.json
|   |   |-- 19faafc142134d3dc0aa8549548ebfc9.json
|   |   |-- 1b929099df8f8367c0f113f68c68a871.json
|   |   |-- 243b63e25b0d08e82027fbf0b7094103.json
|   |   |-- 25cc2e7f9d548f223a9246278faf070c.json
|   |   |-- 26af7bf70e39eac85f359a20b4920807.json
|   |   |-- 27e04b34c24f2d7c3388902bf47bc191.json
|   |   |-- 2803b0255bfe5bf5f408f54dd6a5ebdf.json
|   |   |-- 285381b8a52f7710b9e0179d82b4a776.json
|   |   |-- 28806eafb213df2393e3ef3d29c9a922.json
|   |   |-- 2c1db7e7749854b22f422a5d225e102b.json
|   |   |-- 2dcd8acdd969096e97eb55ec62896489.json
|   |   |-- 3005daaa6e5e09029fb79525aee59066.json
|   |   |-- 34c1f15fef3b30c0e8a395d749f67fa5.json
|   |   |-- 37398d33ce57f4128a0e8b3559cfb2cc.json
|   |   |-- 37a0c4732ddf7e8221c364ac44f8eab9.json
|   |   |-- 3ad0175d40866d90b1225c0c613da2d5.json
|   |   |-- 3d699ad0a0ce096558495da43a75e1cd.json
|   |   |-- 417ff21e30bb57b5ec31ad3136105cf5.json
|   |   |-- 43220ef1070d5563db9a6cbff057c30d.json
|   |   |-- 439afa0dfb8d32f88caf3f3582e980ff.json
|   |   |-- 43bb09189061bb3d1ddf7cdc13e692bf.json
|   |   |-- 455e23e3d31676eea550d174bd7604e9.json
|   |   |-- 45dc8647fbeddcf3be53bf42758ad3d9.json
|   |   |-- 47e0bd1041236d8d41b3e0fdd9985893.json
|   |   |-- 48dc9480d1810864e3cfa442fa8fa2c1.json
|   |   |-- 490eb946727bcda95df528582413cf33.json
|   |   |-- 4b5299816e734798bede1e1c99457160.json
|   |   |-- 4cf929979a93707734c91386aea4dd76.json
|   |   |-- 4cfa49870374bad5e09573165cdf0876.json
|   |   |-- 500ccd2942fc1503b1ec3d09883ab86a.json
|   |   |-- 5300153a44295bd4236b8b824b8927b3.json
|   |   |-- 53c446d2eaac0a3a85e1f30392804ebc.json
|   |   |-- 5514f88636b41bb46efe43eb111e53c6.json
|   |   |-- 558543af59ea1c59071c1ccc38666d8b.json
|   |   |-- 583b34af4a14ddb6afa669e3e97d9c4e.json
|   |   |-- 590b73bba502cc5cd2a19ca7b4858aa1.json
|   |   |-- 5a8d75787882d0189c831561d659b4d8.json
|   |   |-- 5c9bdf615de76e8894d8043f6ab87778.json
|   |   |-- 5f7b31425f2b899f3b76be37115f29f4.json
|   |   |-- 617e06212f183fe36e59035d59dccde3.json
|   |   |-- 61b30408a653e25f1513f845271239a2.json
|   |   |-- 621e4f60e2fcb341c6b76a5c8419c52b.json
|   |   |-- 63881eae44e78a616ed5fbba2e6db17e.json
|   |   |-- 65780f871b85574daa57a56def972d5d.json
|   |   |-- 67625a21ad1975f81d4bb0beef488e21.json
|   |   |-- 67a781ab4eae6134fe8c807568b4dbf3.json
|   |   |-- 6acb15df95c2b6b6ad781bad9f0a68a6.json
|   |   |-- 6ad9e44d655d623ca3cdcc4ab3350d6a.json
|   |   |-- 6c6b93504611e51020bba6766b161853.json
|   |   |-- 6c7c83f2f950852e59f87f7b5bb89a28.json
|   |   |-- 6e2fd79a56eaeade8477595cd9f53520.json
|   |   |-- 705929c5d945bacf199de3fa4bdf204a.json
|   |   |-- 70f1a8479ff6d5fe700365a654960f5f.json
|   |   |-- 72f1f46acbd1f83eb16f692c9792ec4d.json
|   |   |-- 756ed7fc4a2e901b3db3a6b922fd1942.json
|   |   |-- 788ddbc1478e8c1bf59743cbbd96586d.json
|   |   |-- 79a705417debdcb02063526d12e9b644.json
|   |   |-- 79b46006f40ee8db45f11edfefe7ddd3.json
|   |   |-- 7bc8665c204287c362faa780e31297e1.json
|   |   |-- 7c4194ea2040d6d24523756d924d8459.json
|   |   |-- 7c666c837022dbee449ee47930a57bb5.json
|   |   |-- 7da1b364728c627b5f525feddda143a8.json
|   |   |-- 7e35861b26074d13a64cd90dc21b76fb.json
|   |   |-- 7eae05699a78a25e8dc8d7e70fd77dba.json
|   |   |-- 8787906bac9acb3bde98856aa76558c2.json
|   |   |-- 87bf1c48898ecc9151138880796315a2.json
|   |   |-- 8821d39d8b1f3342b2e28464e5bb830c.json
|   |   |-- 8853127c5d1c1e6eb6273cf4a1eb736b.json
|   |   |-- 8c14a916f9a3fa84e80484d1dac3a53f.json
|   |   |-- 8ca51592699aa5ff5b934b20fb75bec5.json
|   |   |-- 91ceb3a6276bceac38896959298b1b51.json
|   |   |-- 93a933f277000065695b97970fe4ad32.json
|   |   |-- 93ae7a22a9154eb36fed65cece074b01.json
|   |   |-- 9558e5568d891482f6755aa3f395b429.json
|   |   |-- 959fdafbad843c84d86565240afd7fb0.json
|   |   |-- 97c1a5f9371babbc560d44f15ba29b5c.json
|   |   |-- 9b1b80a13a352f05ce5754ec3173594b.json
|   |   |-- 9b4f49642a85012ed855a54bd43529a7.json
|   |   |-- 9b8f3de4cab73ac6f7f52694046bddde.json
|   |   |-- 9e3587fc754c1acf77979a5d13e1d11e.json
|   |   |-- a0146a4528f2683a816a3c33837481a8.json
|   |   |-- a095589c0253878accfbf78c585ca6e9.json
|   |   |-- a99aebc4a407374c3115a59550aa77dd.json
|   |   |-- ac36a2bad65dcad7502c0e059d989a82.json
|   |   |-- ac594d193418c8d69366713951eabe7e.json
|   |   |-- aea4979d4ee35240c140f798f6815e60.json
|   |   |-- aeb0716f288e8dd61c6fd952ab3d1e91.json
|   |   |-- af36e46b195974f3444b99f3642cf143.json
|   |   |-- af6e82f9867a10a7130eec127be780fa.json
|   |   |-- b25c38f0a366749876880b4c12e60c6a.json
|   |   |-- b4d9f3b39307fed82596823f7f7b1c41.json
|   |   |-- b556cd9de5d0b56b950eb689d618af2a.json
|   |   |-- b71bb2b342d6130e1fbacf2987bf30dc.json
|   |   |-- b73386714d23ba3f22360133a543144e.json
|   |   |-- b912f0c09dd70782c05d8584343a7ed8.json
|   |   |-- b99f2d4d56a0ba133163e81ca2954675.json
|   |   |-- bab0cfbb60d4392ee7744c1b4e0fc21b.json
|   |   |-- bac7023487098964e570155fd0ab6482.json
|   |   |-- bae5e8573a682c5db3ae256a13ff35f4.json
|   |   |-- bdf4ce1f1707e2bd7b67fcf11e00db4f.json
|   |   |-- bf785f755c39c326fe0178fb5d14e729.json
|   |   |-- c5369e052ef330b720eb92abb784abf0.json
|   |   |-- c80c08412e43a9f416125bc759b861f0.json
|   |   |-- c890cf698f92511ddc42270d546e9a37.json
|   |   |-- c9342fd5a9551f2220a063b41314c3eb.json
|   |   |-- c9c4c4f91069f1c5cfeca6bea60a867a.json
|   |   |-- c9f570632839cb55e44ef3e8e339002a.json
|   |   |-- ce041c4fa9a53b983a72fe8a78a0c39a.json
|   |   |-- d0a83a977d092ad6cdbf63f2fc9d77fd.json
|   |   |-- d0c238cb1ae1ee05f9c1a98a14ee03ae.json
|   |   |-- d11e9819b5eb4422b0d944b2c4101001.json
|   |   |-- d33b951a1733198974ac4d8ed3a16065.json
|   |   |-- d39c09106050b3c4fee6f1dd3d664027.json
|   |   |-- d3eeb14dc44ce07b4ed5024578732c19.json
|   |   |-- d63a714fdcd8f16280effd0d069a1bae.json
|   |   |-- d64ab6ada9089f662ed62769a9529f4e.json
|   |   |-- d67d67eb4128bb0c0425439441214046.json
|   |   |-- d7b669cd48fc0e235b1cf9f19fcd181e.json
|   |   |-- d7c032bc15054c2c0d280b3e419d4662.json
|   |   |-- d97c9197251eb3c2c6f2274ec742d9a7.json
|   |   |-- da6e2f61e61578d7370b32e0fc70182c.json
|   |   |-- da843360b6002d42e88203e9f200a437.json
|   |   |-- dafadc2300ac7e2e28e3e2c399beb0ec.json
|   |   |-- dc34ce2c6c8e3cc2b98c32e2bb908b43.json
|   |   |-- dc7e1ed2f817b032455f51e03f7ee937.json
|   |   |-- dc96a8f7b6594d20bd2237613632418b.json
|   |   |-- dd00257de9ead6149e2d05d3a19c36c0.json
|   |   |-- dee9983a93d9b16d97cb445ade3a1ae5.json
|   |   |-- e08fa4b752599a118bdb09defe3a5bbb.json
|   |   |-- e325a57e2b7b222d8149568cc8d0b448.json
|   |   |-- e39216f0de390fdd2bc8e5477970a64e.json
|   |   |-- e500941839ec5858ba6deda8c258a5f2.json
|   |   |-- e5f589d1ea3fc8f3dc96d62da3341a90.json
|   |   |-- e96576dbe44a83e9f3a6514e8d7d38f6.json
|   |   |-- ea386c573f71d62bb63a4b417e5ea54d.json
|   |   |-- ea7af20a79762e07075cb12c6952b96e.json
|   |   |-- ec8d3c1ca844a7f5766e622fbd2e51b7.json
|   |   |-- ed78e3051122e874244237334d0f7a47.json
|   |   |-- ed9c4cd2d8521c68d3e3b52b8ff2e75d.json
|   |   |-- f4b5d71cc1b043d767b7a49622e2f0e4.json
|   |   |-- f4f06921603e32a38460b0ffd2894297.json
|   |   |-- f64d0b5daf1e36e0957929bc8b35fb1e.json
|   |   |-- f69110e8e4b2253876995f1bb8b00b2f.json
|   |   |-- f6f156e0f858c2d28f568c29b704ae56.json
|   |   |-- fb27b254e46d60cbd350709a99ad0cde.json
|   |   \-- ff49bf28e2c51529730cd49e22e8612d.json
|   |-- cache_manager.py
|   |-- data_models.py
|   |-- main.py
|   |-- ml_models.py
|   |-- model_interfaces.py
|   |-- model_registry.py
|   |-- object_detection.py
|   |-- query_understanding.py
|   |-- requirements.txt
|   |-- result_validator.py
|   |-- sample_missions.py
|   |-- spatial_reasoning.py
|   |-- storage/
|   |   |-- images/
|   |   |   |-- 0aafa0b7-5b82-44cc-a463-a05a5c51a838_0.img
|   |   |   \-- 0db3742b-5c7e-4ca2-b98f-aa13e784274a_0.img
|   |   \-- queries.db
|   |-- storage_manager.py
|   |-- storage_r2.py
|   |-- test_change_pipeline.py
|   |-- test_cross_modal.py
|   |-- test_input_robustness.py
|   |-- test_query_routing.py
|   |-- test_tool_registry.py
|   |-- tiling.py
|   \-- tool_registry.py
|-- docs/
|   |-- DEMO_SCRIPT_230_PRECISE.md
|   |-- DEMO_SCRIPT_2MIN30SEC.md
|   |-- DEPLOYMENT_GUIDE.md
|   |-- FINAL_DEMO_QUERIES.md
|   |-- FINAL_POSTER_CORRECTIONS.md
|   |-- FINAL_PROJECT_AUDIT.md
|   |-- MODEL_AUDIT_FACTS.json
|   |-- MODEL_AUDIT_REPORT.md
|   |-- PROJECT_CONTEXT.md
|   |-- SIH_2026_PRESENTATION_MASTER_DECK.txt
|   |-- SIH_2026_PRESENTATION_VERIFIED_DECK.md
|   |-- VIDEO_PITCH_SCRIPT.md
|   \-- model_choice.md
|-- frontend/
|   |-- .env.local
|   |-- .gitignore
|   |-- AGENTS.md
|   |-- CLAUDE.md
|   |-- README.md
|   |-- eslint.config.mjs
|   |-- next-env.d.ts
|   |-- next.config.ts
|   |-- package-lock.json
|   |-- package.json
|   |-- postcss.config.mjs
|   |-- public/
|   |   |-- apple-touch-icon.png
|   |   |-- asteroid.jpg
|   |   |-- cyborg_frog.jpg
|   |   |-- demo/
|   |   |   |-- bengaluru_cartosat3.png
|   |   |   |-- cyclone_cloud_obscured_optical.png
|   |   |   |-- mumbai_cartosat2s_optical.png
|   |   |   |-- risat1_cband_radar_sar.png
|   |   |   |-- sambhar_lake_liss4.png
|   |   |   |-- uttarakhand_post_event_t2.png
|   |   |   \-- uttarakhand_pre_event_t1.png
|   |   |-- file.svg
|   |   |-- gallery/
|   |   |   |-- photo-1443456066412-3e3ea69ee37c.avif
|   |   |   |-- photo-1446776811953-b23d57bd21aa.avif
|   |   |   |-- photo-1517866184231-7ef94c2ea930.avif
|   |   |   |-- photo-1529788295308-1eace6f67388.avif
|   |   |   |-- photo-1534996858221-380b92700493.avif
|   |   |   |-- photo-1543722530-d2c3201371e7.avif
|   |   |   |-- photo-1558158539-226f4a45f7b3.avif
|   |   |   |-- photo-1614729939124-032f0b56c9ce.avif
|   |   |   |-- photo-1670884307247-d1e905df653d.avif
|   |   |   |-- premium_photo-1677511580659-f5fa0675a547.avif
|   |   |   |-- premium_photo-1690571200236-0f9098fc6ca9.avif
|   |   |   |-- premium_photo-1712039658659-7019cfe912e1.avif
|   |   |   \-- premium_photo-1722018576685-45a415a4ff67.avif
|   |   |-- globe.svg
|   |   |-- icons/
|   |   |   |-- apple-touch-icon.png
|   |   |   |-- icon-192x192.png
|   |   |   |-- icon-512x512.png
|   |   |   |-- icon.svg
|   |   |   \-- maskable-icon-512x512.png
|   |   |-- images/
|   |   |   |-- gslv_mission.webp
|   |   |   \-- pslv_mission.webp
|   |   |-- manifest.json
|   |   |-- next.svg
|   |   |-- planet.jpg
|   |   |-- planets/
|   |   |   |-- callisto.jpg
|   |   |   |-- ceres.jpg
|   |   |   |-- earth.jpg
|   |   |   |-- enceladus.jpg
|   |   |   |-- europa.jpg
|   |   |   |-- ganymede.jpg
|   |   |   |-- io.jpg
|   |   |   |-- jupiter.jpg
|   |   |   |-- mars.jpg
|   |   |   |-- mercury.jpg
|   |   |   |-- moon.jpg
|   |   |   |-- neptune.jpg
|   |   |   |-- pluto.jpg
|   |   |   |-- saturn.jpg
|   |   |   |-- sun.jpg
|   |   |   |-- titan.jpg
|   |   |   |-- triton.jpg
|   |   |   |-- uranus.jpg
|   |   |   |-- venus.jpg
|   |   |   \-- vesta.jpg
|   |   |-- sw.js
|   |   |-- textures/
|   |   |   |-- 2k_earth_clouds.jpg
|   |   |   |-- 2k_earth_daymap.jpg
|   |   |   |-- 2k_earth_normal_map.tif
|   |   |   |-- 2k_earth_specular_map.tif
|   |   |   |-- 2k_jupiter.jpg
|   |   |   |-- 2k_mars.jpg
|   |   |   |-- 2k_mercury.jpg
|   |   |   |-- 2k_neptune.jpg
|   |   |   |-- 2k_saturn.jpg
|   |   |   |-- 2k_saturn_ring_alpha.png
|   |   |   |-- 2k_sun.jpg
|   |   |   |-- 2k_uranus.jpg
|   |   |   \-- 2k_venus_surface.jpg
|   |   |-- vercel.svg
|   |   \-- window.svg
|   |-- src/
|   |   |-- app/
|   |   |   |-- acquire/
|   |   |   |-- dashboard/
|   |   |   |-- faq/
|   |   |   |-- favicon.ico
|   |   |   |-- gallery/
|   |   |   |-- globals.css
|   |   |   |-- layout.tsx
|   |   |   |-- manifest.ts
|   |   |   |-- offline/
|   |   |   |-- page.tsx
|   |   |   \-- query/
|   |   |-- components/
|   |   |   |-- AuthModal.tsx
|   |   |   |-- BackgroundSlideshow.tsx
|   |   |   |-- CookieConsent.tsx
|   |   |   |-- FAQ.tsx
|   |   |   |-- FadeInScroll.tsx
|   |   |   |-- FramerGlobe.tsx
|   |   |   |-- GeoJSONMapOverlay.tsx
|   |   |   |-- GlobeCarousel3d.tsx
|   |   |   |-- MapExplorer.tsx
|   |   |   |-- MissionPresetsModal.tsx
|   |   |   |-- PWAController.tsx
|   |   |   |-- PageTransitionEffect.tsx
|   |   |   |-- RealGlobe.tsx
|   |   |   |-- RocketLaunchSequence.tsx
|   |   |   |-- SmoothScroll.tsx
|   |   |   |-- SphereOrbit.tsx
|   |   |   |-- SwipeSlider.tsx
|   |   |   |-- SystemLoader.module.css
|   |   |   |-- SystemLoader.tsx
|   |   |   |-- TiffPreview.tsx
|   |   |   |-- WebsiteLoader.module.css
|   |   |   \-- WebsiteLoader.tsx
|   |   \-- context/
|   |       \-- AuthContext.tsx
|   |-- tsconfig.json
|   \-- tsconfig.tsbuildinfo
|-- graphify-out/
|   |-- .graphify_labels.json
|   |-- .graphify_labels.json.sig
|   |-- .graphify_root
|   |-- 2026-09-15/
|   |   |-- .graphify_labels.json
|   |   |-- GRAPH_REPORT.md
|   |   |-- graph.json
|   |   \-- manifest.json
|   |-- 2026-09-16/
|   |   |-- .graphify_labels.json
|   |   |-- GRAPH_REPORT.md
|   |   |-- graph.json
|   |   \-- manifest.json
|   |-- 2026-09-17/
|   |   |-- .graphify_labels.json
|   |   |-- GRAPH_REPORT.md
|   |   |-- graph.json
|   |   \-- manifest.json
|   |-- 2026-09-18/
|   |   |-- .graphify_labels.json
|   |   |-- GRAPH_REPORT.md
|   |   |-- graph.json
|   |   \-- manifest.json
|   |-- 2026-09-20/
|   |   |-- .graphify_labels.json
|   |   |-- GRAPH_REPORT.md
|   |   |-- graph.json
|   |   \-- manifest.json
|   |-- 2026-09-21/
|   |   |-- .graphify_labels.json
|   |   |-- GRAPH_REPORT.md
|   |   |-- graph.json
|   |   \-- manifest.json
|   |-- 2026-09-22/
|   |   |-- .graphify_labels.json
|   |   |-- GRAPH_REPORT.md
|   |   |-- graph.json
|   |   \-- manifest.json
|   |-- 2026-09-23/
|   |   |-- .graphify_labels.json
|   |   |-- GRAPH_REPORT.md
|   |   |-- graph.json
|   |   \-- manifest.json
|   |-- 2026-09-24/
|   |   |-- .graphify_labels.json
|   |   |-- GRAPH_REPORT.md
|   |   |-- graph.json
|   |   \-- manifest.json
|   |-- 2026-09-25/
|   |   |-- .graphify_labels.json
|   |   |-- GRAPH_REPORT.md
|   |   |-- graph.json
|   |   \-- manifest.json
|   |-- 2026-10-14/
|   |   |-- .graphify_labels.json
|   |   |-- GRAPH_REPORT.md
|   |   |-- graph.json
|   |   \-- manifest.json
|   |-- GRAPH_REPORT.md
|   |-- cache/
|   |   |-- ast/
|   |   |   \-- v0.9.54-s2/
|   |   |-- last_query_stamp
|   |   |-- semantic/
|   |   |   \-- p5e80268fecd6/
|   |   \-- stat-index.json
|   |-- graph.html
|   |-- graph.json
|   \-- manifest.json
|-- ml_pipeline/
|   |-- README.md
|   |-- _tmp_downloads/
|   |   \-- train.zip
|   |-- benchmark_audit.py
|   |-- cdvqa_evaluation_report.json
|   |-- change_model.py
|   |-- checkpoints/
|   |   |-- best_satsegnet.pth
|   |   |-- satquery_change_net/
|   |   |   |-- change_net.pth
|   |   |   \-- training_report.json
|   |   |-- satquery_optical_sar_net/
|   |   |   |-- optical_sar_net.pth
|   |   |   \-- training_report.json
|   |   |-- satquery_rs_vlm/
|   |   |   |-- config.json
|   |   |   |-- generation_config.json
|   |   |   |-- model.safetensors
|   |   |   |-- processor_config.json
|   |   |   |-- tokenizer.json
|   |   |   |-- tokenizer_config.json
|   |   |   \-- training_report.json
|   |   |-- satsegnet.onnx
|   |   |-- satsegnet_quantized.onnx
|   |   \-- training_metrics.json
|   |-- cross_modal_model.py
|   |-- data/
|   |   |-- __init__.py
|   |   |-- bigearthnet_loader.py
|   |   |-- cdvqa_loader.py
|   |   |-- rsvqa_loader.py
|   |   \-- vrsbench_loader.py
|   |-- data_sanity.py
|   |-- dataset/
|   |   |-- dataset_manifest.json
|   |   |-- test/
|   |   |   |-- images/
|   |   |   |-- masks/
|   |   |   \-- test_qa_pairs.json
|   |   |-- train/
|   |   |   |-- images/
|   |   |   |-- masks/
|   |   |   \-- train_qa_pairs.json
|   |   \-- val/
|   |       |-- images/
|   |       |-- masks/
|   |       \-- val_qa_pairs.json
|   |-- dataset_cache/
|   |   |-- bigearthnet/
|   |   |   |-- dataset_metadata.json
|   |   |   |-- test/
|   |   |   |-- train/
|   |   |   \-- val/
|   |   |-- cdvqa/
|   |   |   |-- dataset_metadata.json
|   |   |   |-- test/
|   |   |   |-- train/
|   |   |   \-- val/
|   |   |-- rsvqa/
|   |   |   |-- dataset_metadata.json
|   |   |   |-- test/
|   |   |   |-- train/
|   |   |   \-- val/
|   |   \-- vrsbench/
|   |       |-- dataset_metadata.json
|   |       |-- test/
|   |       |-- train/
|   |       \-- val/
|   |-- datasets.py
|   |-- download_real_dataset.py
|   |-- eval_change.py
|   |-- eval_cross_modal.py
|   |-- eval_isro.py
|   |-- eval_routing.py
|   |-- eval_single_image.py
|   |-- evaluate.py
|   |-- evaluation_results.json
|   |-- export_onnx.py
|   |-- isro_evaluation_set.json
|   |-- profile_latency.py
|   |-- requirements.txt
|   |-- sat_seg_model.py
|   |-- satellite_data_engine.py
|   |-- satquery_finetuned_model/
|   |   |-- checkpoint-100/
|   |   |   |-- config.json
|   |   |   |-- generation_config.json
|   |   |   |-- model.safetensors
|   |   |   |-- optimizer.pt
|   |   |   |-- rng_state.pth
|   |   |   |-- scheduler.pt
|   |   |   |-- trainer_state.json
|   |   |   \-- training_args.bin
|   |   |-- checkpoint-95/
|   |   |   |-- config.json
|   |   |   |-- generation_config.json
|   |   |   |-- model.safetensors
|   |   |   |-- optimizer.pt
|   |   |   |-- rng_state.pth
|   |   |   |-- scheduler.pt
|   |   |   |-- trainer_state.json
|   |   |   \-- training_args.bin
|   |   |-- config.json
|   |   |-- generation_config.json
|   |   |-- model.safetensors
|   |   |-- processor_config.json
|   |   |-- tokenizer.json
|   |   |-- tokenizer_config.json
|   |   \-- training_args.bin
|   |-- test_predictions_montage.png
|   |-- train_change_detector.py
|   |-- train_cross_modal.py
|   |-- train_rs_vlm.py
|   |-- train_segmentation.py
|   |-- train_vqa.py
|   \-- vrsbench_evaluation_report.json
|-- reports/
|   |-- change.json
|   |-- change.md
|   |-- cross_modal.json
|   |-- cross_modal.md
|   |-- data_sanity.md
|   |-- final_matrix.md
|   |-- isro_evaluation.json
|   |-- latency.json
|   |-- latency.md
|   |-- phase_1.json
|   |-- phase_1.md
|   |-- phase_2.json
|   |-- phase_2.md
|   |-- phase_3.json
|   |-- phase_3.md
|   |-- phase_4.json
|   |-- phase_4.md
|   |-- phase_5.json
|   |-- phase_5.md
|   |-- phase_6.json
|   |-- phase_6.md
|   |-- phase_7.json
|   |-- phase_7.md
|   |-- phase_8.json
|   |-- phase_8.md
|   |-- phase_9.json
|   |-- phase_9.md
|   |-- probe_results.json
|   |-- routing.json
|   |-- routing.md
|   |-- single_image.json
|   \-- single_image.md
|-- run_all.ps1
|-- run_backend.ps1
|-- scratch/
|   |-- build_verification_dump.py
|   |-- check_confidence_grep.py
|   |-- routing_samples.json
|   |-- run_behaviour_probes.py
|   \-- search_suspicious_words.py
|-- scripts/
|   |-- audit_claims.py
|   \-- generate_routing_benchmark.py
\-- tests/
    \-- routing_cases.jsonl
```

### 1.7 Weight & Checkpoint File Sizes
```
./.huggingface_cache/hub/models--Salesforce--blip-image-captioning-base/.no_exist/82a37760796d32b1411fe092ab5d4e227313294b/model.safetensors :     0.0000 MB (0 bytes)
./.huggingface_cache/hub/models--Salesforce--blip-image-captioning-base/snapshots/4c26dfece70e02028433dd192458a54b390b85d2/model.safetensors :   943.8718 MB (989,721,336 bytes)
./.huggingface_cache/hub/models--Salesforce--blip-image-captioning-base/snapshots/82a37760796d32b1411fe092ab5d4e227313294b/pytorch_model.bin :   943.9667 MB (989,820,849 bytes)
./.huggingface_cache/hub/models--Salesforce--blip-vqa-base/snapshots/787b3d35d57e49572baabd22884b3d5a05acf072/model.safetensors :  1467.5146 MB (1,538,800,584 bytes)
./.huggingface_cache/hub/models--google--owlvit-base-patch32/snapshots/cbc355fb364588351c5d51c7f74465e8e7ec6f72/model.safetensors :   584.5870 MB (612,983,940 bytes)
./ml_pipeline/checkpoints/best_satsegnet.pth                           :     5.6013 MB (5,873,379 bytes)
./ml_pipeline/checkpoints/satquery_change_net/change_net.pth           :     6.5638 MB (6,882,677 bytes)
./ml_pipeline/checkpoints/satquery_change_net/training_report.json     :     0.0006 MB (599 bytes)
./ml_pipeline/checkpoints/satquery_optical_sar_net/optical_sar_net.pth :     3.2871 MB (3,446,729 bytes)
./ml_pipeline/checkpoints/satquery_optical_sar_net/training_report.json :     0.0006 MB (601 bytes)
./ml_pipeline/checkpoints/satquery_rs_vlm/config.json                  :     0.0024 MB (2,508 bytes)
./ml_pipeline/checkpoints/satquery_rs_vlm/generation_config.json       :     0.0007 MB (725 bytes)
./ml_pipeline/checkpoints/satquery_rs_vlm/model.safetensors            :  1378.0806 MB (1,445,022,200 bytes)
./ml_pipeline/checkpoints/satquery_rs_vlm/processor_config.json        :     0.0005 MB (572 bytes)
./ml_pipeline/checkpoints/satquery_rs_vlm/tokenizer.json               :     0.6785 MB (711,493 bytes)
./ml_pipeline/checkpoints/satquery_rs_vlm/tokenizer_config.json        :     0.0005 MB (530 bytes)
./ml_pipeline/checkpoints/satquery_rs_vlm/training_report.json         :     0.0013 MB (1,354 bytes)
./ml_pipeline/checkpoints/satsegnet.onnx                               :     1.8483 MB (1,938,115 bytes)
./ml_pipeline/checkpoints/satsegnet_quantized.onnx                     :     0.6220 MB (652,186 bytes)
./ml_pipeline/checkpoints/training_metrics.json                        :     0.0006 MB (673 bytes)
./ml_pipeline/satquery_finetuned_model/checkpoint-100/model.safetensors :  1378.0806 MB (1,445,022,200 bytes)
./ml_pipeline/satquery_finetuned_model/checkpoint-100/optimizer.pt     :  2756.5767 MB (2,890,480,156 bytes)
./ml_pipeline/satquery_finetuned_model/checkpoint-100/rng_state.pth    :     0.0138 MB (14,455 bytes)
./ml_pipeline/satquery_finetuned_model/checkpoint-100/scheduler.pt     :     0.0014 MB (1,465 bytes)
./ml_pipeline/satquery_finetuned_model/checkpoint-100/training_args.bin :     0.0050 MB (5,265 bytes)
./ml_pipeline/satquery_finetuned_model/checkpoint-95/model.safetensors :  1378.0806 MB (1,445,022,200 bytes)
./ml_pipeline/satquery_finetuned_model/checkpoint-95/optimizer.pt      :  2756.5767 MB (2,890,480,156 bytes)
./ml_pipeline/satquery_finetuned_model/checkpoint-95/rng_state.pth     :     0.0138 MB (14,455 bytes)
./ml_pipeline/satquery_finetuned_model/checkpoint-95/scheduler.pt      :     0.0014 MB (1,465 bytes)
./ml_pipeline/satquery_finetuned_model/checkpoint-95/training_args.bin :     0.0050 MB (5,265 bytes)
./ml_pipeline/satquery_finetuned_model/model.safetensors               :  1378.0806 MB (1,445,022,200 bytes)
./ml_pipeline/satquery_finetuned_model/training_args.bin               :     0.0050 MB (5,265 bytes)
```

## 2. MODELS ACTUALLY LOADED

### 2.1 Grep for Model Loading Invocations

Command: `git grep -En "from_pretrained|load_state_dict|InferenceSession|torch\.load" -- backend/ ml_pipeline/`
```
backend/ml_models.py:49:                self.vqa_processor = BlipProcessor.from_pretrained(finetuned_path, local_files_only=True)
backend/ml_models.py:50:                self.vqa_model = BlipForQuestionAnswering.from_pretrained(finetuned_path, local_files_only=True).to(self.device)
backend/ml_models.py:55:                self.vqa_processor = BlipProcessor.from_pretrained(model_id, local_files_only=True)
backend/ml_models.py:56:                self.vqa_model = BlipForQuestionAnswering.from_pretrained(model_id, local_files_only=True).to(self.device)
backend/ml_models.py:63:                self.vqa_processor = BlipProcessor.from_pretrained(model_id, local_files_only=True)
backend/ml_models.py:64:                self.vqa_model = BlipForQuestionAnswering.from_pretrained(model_id, local_files_only=True).to(self.device)
backend/ml_models.py:94:                session = ort.InferenceSession(target_onnx, providers=["CPUExecutionProvider"])
backend/ml_models.py:124:                ckpt = torch.load(ckpt_path, map_location=self.device)
backend/ml_models.py:126:                seg_model.load_state_dict(state_dict)
backend/ml_models.py:150:                ckpt = torch.load(ckpt_path, map_location=self.device)
backend/ml_models.py:152:                change_model.load_state_dict(state_dict)
backend/ml_models.py:177:                ckpt = torch.load(ckpt_path, map_location=self.device)
backend/ml_models.py:179:                model.load_state_dict(state_dict)
ml_pipeline/benchmark_audit.py:53:    ckpt = torch.load(checkpoint_path, map_location=device)
ml_pipeline/benchmark_audit.py:54:    model.load_state_dict(ckpt["model_state_dict"])
ml_pipeline/evaluate.py:11:    processor = BlipProcessor.from_pretrained(args.model_path)
ml_pipeline/evaluate.py:12:    model = BlipForQuestionAnswering.from_pretrained(args.model_path)
ml_pipeline/export_onnx.py:28:    ckpt = torch.load(checkpoint_path, map_location="cpu")
ml_pipeline/export_onnx.py:30:    model.load_state_dict(state_dict)
ml_pipeline/export_onnx.py:60:        session = ort.InferenceSession(output_onnx_path, providers=["CPUExecutionProvider"])
ml_pipeline/train_vqa.py:10:    processor = BlipProcessor.from_pretrained(model_id)
ml_pipeline/train_vqa.py:11:    model = BlipForQuestionAnswering.from_pretrained(model_id)
```

### 2.2 Models Loadable at Runtime by Backend


| Model Family | Implementation Class | Checkpoint Path | Exists on Disk? | Exact Param Count | Toggle Env Var | Fallback Behavior if Missing / Disabled |
| :--- | :--- | :--- | :---: | :---: | :--- | :--- |
| **VQA / VLM Foundation** | `BlipForQuestionAnswering` + `BlipProcessor` | `ml_pipeline/satquery_finetuned_model/` (or `ml_pipeline/checkpoints/satquery_rs_vlm/`) | **YES** | **361,230,140** | `ENABLE_ML_MODELS=true` | Falls back to `BLIP-VQA-Base` cache; if that fails or `ENABLE_ML_MODELS!=true`, falls back to `fallback_heuristic (OpenCV Spectral Discriminator)` |
| **Semantic Segmentation** | `ONNXSatSegNetRunner` (primary) / `SatSegNet` PyTorch (secondary) | `ml_pipeline/checkpoints/satsegnet_quantized.onnx` (`satsegnet.onnx` or `best_satsegnet.pth`) | **YES** | **482,822** | None (auto-detects files) | Falls back to `fallback_heuristic (OpenCV Spectral Discriminator)` |
| **Bi-Temporal Change** | `SatQueryChangeNet` (Siamese difference network) | `ml_pipeline/checkpoints/satquery_change_net/change_net.pth` | **YES** | **1,709,873** | None (auto-detects files) | Falls back to `fallback_heuristic (SSIM Structural Difference Engine)` |
| **Optical-SAR Fusion** | `SatQueryOpticalSARNet` (Dual-encoder fusion network) | `ml_pipeline/checkpoints/satquery_optical_sar_net/optical_sar_net.pth` | **YES** | **849,603** | None (auto-detects files) | Falls back to `fallback_heuristic (IHS Cross-Modal Fusion Engine)` |
| **Object Localization** | `connectedComponentsWithStats` contour delineation / bounding box | In-memory OpenCV algorithmic routine | **N/A** (algorithmic) | **0** (heuristic) | None | Emits bounding box coordinates via OpenCV connected components |

### 2.3 Fine-Tuned Vision-Language Model Checkpoint Path

- **Primary Fine-Tuned Directory:** `d:\sih26167\ml_pipeline\satquery_finetuned_model`
  - Contains `model.safetensors` (1,378.08 MB), `config.json`, `generation_config.json`, `processor_config.json`, `tokenizer.json`, `tokenizer_config.json`, plus checkpoint subdirectories `checkpoint-95/` and `checkpoint-100/`.
- **Secondary Export Directory:** `d:\sih26167\ml_pipeline\checkpoints\satquery_rs_vlm`
  - Contains `model.safetensors` (1,378.08 MB), `config.json`, and `training_report.json`.

### 2.4 Live Endpoint Outputs: GET /health and GET /api/preflight

#### GET /health (from live running backend http://127.0.0.1:8000/health)
```json
{
  "status": "online",
  "service": "SatQuery AI Backend",
  "problem_statement": "ISRO / SAC \u2014 PS 26167",
  "version": "1.0.0",
  "uptime_seconds": 3859.8,
  "runtime": {
    "device": "cpu",
    "transformers_offline": true,
    "enable_ml_models": true,
    "cache_backend": "upstash_redis_rest",
    "storage_backend": "sqlite",
    "object_storage": "inline_base64_fallback"
  },
  "models": {
    "vlm_foundation": {
      "loaded_name": "SatQuery-RS-VLM-v1.0 (Adapted on BigEarthNet.txt, RSVQA, VRSBench)",
      "model_class": "BlipForQuestionAnswering",
      "checkpoint_path": "D:\\sih26167\\ml_pipeline\\satquery_finetuned_model",
      "status": "ready"
    },
    "segmentation": {
      "loaded_name": "SatSegNet-v1.0 (ONNX INT8 Quantized)",
      "architecture": "SatSegNet (From-scratch 6-class Multi-Scale U-Net, 482,822 parameters)",
      "checkpoint_path": "D:\\sih26167\\ml_pipeline\\checkpoints\\best_satsegnet.pth",
      "classes": [
        "Background",
        "Water",
        "Forest/Vegetation",
        "Bare Soil",
        "Urban/Built-up",
        "Cloud/Haze"
      ],
      "status": "ready"
    },
    "change_detection": {
      "loaded_name": "SatQueryChangeNet-v1.0 (Siamese Difference Network, Val IoU: 88.1%)",
      "architecture": "SatQueryChangeNet (Siamese Feature Difference Network, 1,709,873 parameters)",
      "checkpoint_path": "D:\\sih26167\\ml_pipeline\\checkpoints\\satquery_change_net\\change_net.pth",
      "val_iou": 0.8807,
      "status": "ready"
    },
    "sar_fusion": {
      "loaded_name": "SatQueryOpticalSARNet-v1.0 (Optical-SAR Fusion Network, BigEarthNet trained, Val IoU: 34.4%)",
      "architecture": "SatQueryOpticalSARNet (Optical-SAR Dual-Encoder Feature Fusion, 849,603 parameters)",
      "checkpoint_path": "D:\\sih26167\\ml_pipeline\\checkpoints\\satquery_optical_sar_net\\optical_sar_net.pth",
      "val_iou": 0.3442,
      "status": "ready"
    },
    "spatial_grounding": {
      "algorithm": "Connected-component subpixel contour delineation (RFC 7946 GeoJSON)",
      "status": "ready"
    }
  },
  "gis_integration": {
    "bhuvan_wms": "enabled (bhuvan-vec2.nrsc.gov.in)",
    "geojson_export": "enabled (EPSG:4326 / RFC 7946)"
  }
}
```

#### GET /api/preflight
```
HTTP 405 Method Not Allowed
{"detail":"Method Not Allowed"}
```
*(Note: `/api/preflight` is declared as `@app.post("/api/preflight")` in `backend/main.py:166`; standard HTTP GET returns 405 Method Not Allowed as expected.)*

## 3. TRAINING EVIDENCE

### 3.1 SatSegNet (Semantic Segmentation)

- **Training script path:** `ml_pipeline/train_segmentation.py`
- **Execution command:** `python ml_pipeline/train_segmentation.py --epochs 10 --batch_size 16 --lr 0.001`
- **Dataset name & source:** `ml_pipeline/dataset/` generated by `ml_pipeline/satellite_data_engine.py` using local images from `.user_uploaded` source folder (`C:\Users\athar\.gemini\antigravity-ide\brain\01bb0c37-cc1e-49f6-b398-e17a23eda90b\.user_uploaded`).
- **Sample counts:** 240 train images, 30 val images, 30 test images (all 256x256).
- **Data nature:** **SYNTHETIC / SELF-LABELLED PSEUDO-LABELS**. `ml_pipeline/satellite_data_engine.py:extract_dense_ground_truth()` used heuristic OpenCV rules (HSV color thresholds, Canny edges, local blur) to assign pseudo-ground-truth class masks.
- **Split method:** Augmented spatial crops were extracted from the source images and then shuffled randomly with `random.shuffle(all_pairs)` across train (80%), val (10%), and test (10%). **Data leakage alert:** crops originating from the same source images exist across both train and test splits.
- **Hyperparameters:** Epochs: 10, Batch size: 16, Optimizer: Adam (lr=0.001), Seed: 42, Hardware: 12-core host CPU, Wall-clock time: 90.24 seconds.
- **Test split usage:** The test split was evaluated post-training; model selection was governed by validation mIoU (`best_epoch: 9`, `best_miou: 0.726`).
- **Full contents of `ml_pipeline/checkpoints/training_metrics.json`:**
```json

{
  "train_loss": [
    1.0585,
    0.8461,
    0.7324,
    0.6921,
    0.6099,
    0.6167,
    0.5765,
    0.5381,
    0.4816,
    0.4653
  ],
  "val_loss": [
    1.123,
    0.8178,
    0.8186,
    0.6873,
    0.6812,
    0.5788,
    0.6243,
    0.5049,
    0.4722,
    0.4703
  ],
  "val_miou": [
    0.3199,
    0.4365,
    0.3804,
    0.5042,
    0.4656,
    0.6399,
    0.6606,
    0.6924,
    0.726,
    0.7198
  ],
  "epochs": [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10
  ],
  "best_epoch": 9,
  "best_miou": 0.726,
  "total_training_time_seconds": 90.24,
  "final_best_miou": 0.726
}

```

### 3.2 SatQueryChangeNet (Bi-Temporal Change Detection)

- **Training script path:** `ml_pipeline/train_change_detector.py`
- **Execution command:** `python ml_pipeline/train_change_detector.py --epochs 3 --batch_size 8 --lr 0.0005`
- **Dataset name & source:** `ml_pipeline/dataset_cache/cdvqa` (70 train, 15 val, 15 test pairs).
- **Data nature:** **SYNTHETIC / GENERATED SAMPLE PATCHES**. Generated by `ml_pipeline/data/cdvqa_loader.py:create_sample_cdvqa_subset()` using `np.random.RandomState(3000 + idx)` with Gaussian noise and simulated threshold change masks.
- **Hyperparameters:** Epochs: 3, Batch size: 8, Optimizer: Adam (lr=0.0005), Seed: 42, Hardware: 12-core host CPU, Wall-clock time: 162.03 seconds.
- **Test split usage:** Evaluated in `ml_pipeline/eval_change.py` on the 15 test pairs.
- **Full contents of `ml_pipeline/checkpoints/satquery_change_net/training_report.json`:**
```json

{
  "model_name": "SatQueryChangeNet-v1.0",
  "reference": "arXiv:2112.06343",
  "parameters": 1709873,
  "device": "cpu",
  "epochs": 3,
  "total_time_seconds": 162.03,
  "final_val_loss": 0.9449,
  "final_val_iou": 0.8807,
  "history": [
    {
      "epoch": 1,
      "train_loss": 1.064,
      "val_loss": 1.1238,
      "val_iou": 0.9212
    },
    {
      "epoch": 2,
      "train_loss": 0.9891,
      "val_loss": 1.0228,
      "val_iou": 0.8824
    },
    {
      "epoch": 3,
      "train_loss": 0.9397,
      "val_loss": 0.9449,
      "val_iou": 0.8807
    }
  ]
}

```

### 3.3 SatQueryOpticalSARNet (Optical-SAR Fusion)

- **Training script path:** `ml_pipeline/train_cross_modal.py`
- **Execution command:** `python ml_pipeline/train_cross_modal.py --epochs 3 --batch_size 8 --lr 0.0005`
- **Dataset name & source:** `ml_pipeline/dataset_cache/bigearthnet` (70 train, 15 val, 15 test pairs).
- **Data nature:** **SYNTHETIC / GENERATED SAMPLE PATCHES**. Generated by `ml_pipeline/data/bigearthnet_loader.py:create_sample_bigearthnet_subset()` using `np.random.RandomState(42 + idx)` for RGB patches and `rng.uniform(-25.0, 0.0)` for S1 SAR decibels.
- **Hyperparameters:** Epochs: 3, Batch size: 8, Optimizer: Adam (lr=0.0005), Seed: 42, Hardware: 12-core host CPU, Wall-clock time: 91.11 seconds.
- **Test split usage:** Evaluated in `ml_pipeline/eval_cross_modal.py` on the 15 test pairs.
- **Full contents of `ml_pipeline/checkpoints/satquery_optical_sar_net/training_report.json`:**
```json

{
  "model_name": "SatQueryOpticalSARNet-v1.0",
  "reference": "arXiv:2603.29630",
  "parameters": 849603,
  "device": "cpu",
  "epochs": 3,
  "total_time_seconds": 91.11,
  "final_val_loss": 1.1046,
  "final_val_iou": 0.3442,
  "history": [
    {
      "epoch": 1,
      "train_loss": 1.3646,
      "val_loss": 1.3953,
      "val_iou": 0.207
    },
    {
      "epoch": 2,
      "train_loss": 1.2933,
      "val_loss": 1.2776,
      "val_iou": 0.2541
    },
    {
      "epoch": 3,
      "train_loss": 1.1952,
      "val_loss": 1.1046,
      "val_iou": 0.3442
    }
  ]
}

```

### 3.4 SatQuery-RS-VLM-v1.0 (Vision-Language Model)

- **Training script path:** `ml_pipeline/train_rs_vlm.py` (and `ml_pipeline/train_vqa.py`)
- **Execution command:** `python ml_pipeline/train_rs_vlm.py --lr 2e-5 --batch_size 2`
- **Dataset name & source:** Subsets from `ml_pipeline/dataset_cache/`:
  - Stage 1: BigEarthNet.txt (3 samples)
  - Stage 2: RSVQA + VRSBench VQA (6 samples)
  - Stage 3: VRSBench Grounding (3 samples)
  - **Total training samples:** **12 samples total**.
- **Data nature:** Synthetic/simulated sample pairs generated by `ml_pipeline/data/*_loader.py`.
- **Hyperparameters:** Epochs: 1 per stage, Batch size: 2, Learning rate: 2e-05, Optimizer: AdamW, Seed: 42, Hardware: CPU (Host), Wall-clock time: 67.05 seconds.
- **Test split usage:** Evaluated in `ml_pipeline/eval_single_image.py`.
- **Full contents of `ml_pipeline/checkpoints/satquery_rs_vlm/training_report.json`:**
```json

{
  "model_name": "SatQuery-RS-VLM-v1.0",
  "architecture": "BLIP-VQA RS-Adapted (ViT-B/16 + 12-layer Cross-Attention Decoder)",
  "base_model": "Salesforce/blip-vqa-base",
  "total_parameters": 361230140,
  "trainable_parameters": 361230140,
  "training_stages": [
    {
      "stage": 1,
      "name": "Image-Text Alignment & Captioning",
      "dataset": "BigEarthNet.txt (arXiv:2603.29630)",
      "samples": 3
    },
    {
      "stage": 2,
      "name": "RS Visual Question Answering",
      "dataset": "RSVQA (arXiv:2003.07333) + VRSBench VQA",
      "samples": 6
    },
    {
      "stage": 3,
      "name": "Text-Guided Grounding",
      "dataset": "VRSBench Grounding (arXiv:2406.12384)",
      "samples": 3
    }
  ],
  "random_seed": 42,
  "learning_rate": 2e-05,
  "epochs_per_stage": 1,
  "hardware": {
    "device": "cpu",
    "device_name": "CPU (Host)"
  },
  "wall_time_seconds": 67.05,
  "loss_trajectory": [
    {
      "stage": 1,
      "epoch": 1,
      "dataset": "BigEarthNet.txt",
      "loss": 9.2166
    },
    {
      "stage": 2,
      "epoch": 1,
      "dataset": "RSVQA + VRSBench VQA",
      "loss": 1.5982
    },
    {
      "stage": 3,
      "epoch": 1,
      "dataset": "VRSBench Grounding",
      "loss": 8.752
    }
  ],
  "date_executed": "2026-10-14 15:48:37"
}

```

## 4. EVALUATION EVIDENCE

### 4.1 77.82% mIoU (SatSegNet)

- **Producing script:** `NOT FOUND (No script outputs 77.82%).`

- **Command:** `NOT RUN (Synthetically cited number).`

- **Backing JSON File Status:** NONE in reports/*.json. (The actual evaluation file ml_pipeline/evaluation_results.json states 80.22% on 30 test samples; training_metrics.json states 72.60% validation mIoU).

- **Dataset, Split & Samples:** ml_pipeline/dataset/test (30 samples, 491,520 pixels), self-labelled pseudo masks from local images.

- **Train/Test Data Overlap:** YES — random crop augmentation from same source images shuffled into train and test.

- **File Timestamp:** `N/A (Number appears only in README.md, final_matrix.md, and presentation deck).`

### 4.2 71.30% mIoU (ChangeNet)

- **Producing script:** `ml_pipeline/eval_change.py`

- **Command:** `python ml_pipeline/eval_change.py`

- **Backing JSON File Status:** DISCREPANCY. reports/change.json contains 'mean_iou_pct: 15.16%' (not 71.30%). The training report training_report.json states 'final_val_iou: 0.8807' (88.07%). The 71.30% figure has NO backing file in reports/.

- **Dataset, Split & Samples:** ml_pipeline/dataset_cache/cdvqa/test (15 sample pairs generated by cdvqa_loader.py).

- **Train/Test Data Overlap:** Separate index range from train (train: 0-69, test: 0-14 generated separately).

- **File Timestamp:** `reports/change.json: 2026-10-14T16:46:47.588064`

### 4.3 75.80% mIoU (OpticalSARNet)

- **Producing script:** `ml_pipeline/eval_cross_modal.py`

- **Command:** `python ml_pipeline/eval_cross_modal.py`

- **Backing JSON File Status:** DISCREPANCY. reports/cross_modal.json contains 'overall_mean_iou_pct: 24.93%' (not 75.80%). The training report states 'final_val_iou: 0.3442' (34.42%). The 75.80% figure has NO backing file in reports/.

- **Dataset, Split & Samples:** ml_pipeline/dataset_cache/bigearthnet/test (15 sample pairs generated by bigearthnet_loader.py).

- **Train/Test Data Overlap:** Generated separately.

- **File Timestamp:** `reports/cross_modal.json: 2026-10-14T17:10:03.871167`

### 4.4 91.02% Routing Accuracy & 99.31% Valid Sequence Rate

- **Producing script:** `ml_pipeline/eval_routing.py`

- **Command:** `python ml_pipeline/eval_routing.py`

- **Backing JSON File Status:** reports/routing.json (Exact match: task_classification_accuracy_pct: 91.02%, valid_sequence_rate_pct: 99.31%).

- **Dataset, Split & Samples:** tests/routing_cases.jsonl (167 total cases: 145 valid, 22 adversarial).

- **Train/Test Data Overlap:** N/A — Evaluation of backend rule and intent router against hand-scripted benchmark suite.

- **File Timestamp:** `reports/routing.json: 2026-10-14T17:46:13.138345`

### 4.5 9.68 ms Latency

- **Producing script:** `NOT FOUND in reports/latency.json.`

- **Command:** `NOT RUN for 9.68 ms. (reports/latency.json reports real tool latencies: optical_sar_fusion P50=7.07 ms, single_image_grounding P50=13.24 ms, land_cover_segmentation P50=13.76 ms, bi_temporal_change P50=51.76 ms).`

- **Backing JSON File Status:** NONE in reports/latency.json. The 9.68 ms was cited in documentation from an uncommitted ONNX 128px benchmark.

- **Dataset, Split & Samples:** In-memory synthetic tensors (15 iterations per tool).

- **Train/Test Data Overlap:** N/A

- **File Timestamp:** `reports/latency.json: 2026-10-14T18:00:23.174261`

### 4.6 76.67% VRSBench VQA Accuracy

- **Producing script:** `ml_pipeline/eval_single_image.py`

- **Command:** `python ml_pipeline/eval_single_image.py`

- **Backing JSON File Status:** reports/single_image.json (Exact match: vrsbench_vqa.accuracy_pct: 76.67%).

- **Dataset, Split & Samples:** ml_pipeline/dataset_cache/vrsbench/test (30 VQA questions generated by vrsbench_loader.py).

- **Train/Test Data Overlap:** Generated sample split.

- **File Timestamp:** `reports/single_image.json: 2026-10-14T16:06:15.067995`

### 4.7 100.0% CDVQA Existence Accuracy

- **Producing script:** `ml_pipeline/eval_change.py`

- **Command:** `python ml_pipeline/eval_change.py`

- **Backing JSON File Status:** reports/change.json (Exact match: existence exact_accuracy_pct: 100.0%, 6/6 questions correct).

- **Dataset, Split & Samples:** ml_pipeline/dataset_cache/cdvqa/test (6 existence questions generated by cdvqa_loader.py).

- **Train/Test Data Overlap:** Generated sample split.

- **File Timestamp:** `reports/change.json: 2026-10-14T16:46:47.588064`

### 4.8 100.0% Guardrail True Reject Rate

- **Producing script:** `ml_pipeline/eval_cross_modal.py`

- **Command:** `python ml_pipeline/eval_cross_modal.py`

- **Backing JSON File Status:** reports/cross_modal.json (Exact match: guardrail_verification_rate_pct: 100.0%).

- **Dataset, Split & Samples:** 2 test evaluations (Mumbai co-registered vs Bengaluru-Mumbai disparate pair).

- **Train/Test Data Overlap:** Demo images from frontend/public/demo/.

- **File Timestamp:** `reports/cross_modal.json: 2026-10-14T17:10:03.871167`

### 4.9 Full Content of Key Reports (First 150 Lines Each)

#### `reports/routing.json`
```json
{
  "benchmark": "SatQuery-Agentic-Routing-v1.0 (tests/routing_cases.jsonl)",
  "timestamp": "2026-10-14T17:46:13+05:30",
  "total_cases_evaluated": 167,
  "valid_cases_count": 145,
  "adversarial_cases_count": 22,
  "metrics": {
    "task_classification_accuracy_pct": 91.02,
    "tool_selection_accuracy_pct": 91.02,
    "valid_sequence_rate_pct": 99.31,
    "invalid_call_rejection_rate_pct": 90.91,
    "structured_error_code_match_rate_pct": 86.36,
    "overall_accuracy_pct": 90.42
  },
  "per_category_metrics": {
    "single_image_vqa": {
      "total": 30,
      "accuracy_pct": 93.33,
      "task_acc_pct": 93.33,
      "tool_acc_pct": 93.33
    },
    "single_image_captioning": {
      "total": 20,
      "accuracy_pct": 70.0,
      "task_acc_pct": 70.0,
      "tool_acc_pct": 70.0
    },
    "single_image_grounding": {
      "total": 25,
      "accuracy_pct": 88.0,
      "task_acc_pct": 88.0,
      "tool_acc_pct": 88.0
    },
    "land_cover": {
      "total": 25,
      "accuracy_pct": 96.0,
      "task_acc_pct": 96.0,
      "tool_acc_pct": 96.0
    },
    "change": {
      "total": 25,
      "accuracy_pct": 100.0,
      "task_acc_pct": 100.0,
      "tool_acc_pct": 100.0
    },
    "cross_modal": {
      "total": 20,
      "accuracy_pct": 100.0,
      "task_acc_pct": 100.0,
      "tool_acc_pct": 100.0
    },
    "adversarial_ood": {
      "total": 8,
      "accuracy_pct": 75.0,
      "task_acc_pct": 75.0,
      "tool_acc_pct": 75.0
    },
    "adversarial_count": {
      "total": 6,
      "accuracy_pct": 83.33,
      "task_acc_pct": 83.33,
      "tool_acc_pct": 83.33
    },
    "adversarial_modality": {
      "total": 4,
      "accuracy_pct": 100.0,
      "task_acc_pct": 100.0,
      "tool_acc_pct": 100.0
    },
    "adversarial_format": {
      "total": 1,
      "accuracy_pct": 100.0,
      "task_acc_pct": 100.0,
      "tool_acc_pct": 100.0
    },
    "adversarial_param": {
      "total": 3,
      "accuracy_pct": 66.67,
      "task_acc_pct": 100.0,
      "tool_acc_pct": 100.0
    }
  },
  "latency_by_task": {
    "SINGLE_IMAGE_VQA": {
      "count": 40,
      "mean_ms": 436.9,
      "p50_ms": 407.59,
      "p95_ms": 606.27
    },
    "SINGLE_IMAGE_GROUNDING": {
      "count": 24,
      "mean_ms": 339.43,
      "p50_ms": 357.44,
      "p95_ms": 497.8
    },
    "CHANGE_ANALYSIS": {
      "count": 34,
      "mean_ms": 1159.65,
      "p50_ms": 1513.85,
      "p95_ms": 1658.41
    },
    "SINGLE_IMAGE_CAPTIONING": {
      "count": 14,
      "mean_ms": 445.45,
      "p50_ms": 405.78,
      "p95_ms": 600.62
    },
    "LAND_COVER_SEGMENTATION": {
      "count": 24,
      "mean_ms": 424.8,
      "p50_ms": 414.91,
      "p95_ms": 575.42
    },
    "CROSS_MODAL_EXTRACTION": {
      "count": 23,
      "mean_ms": 144.55,
      "p50_ms": 153.46,
      "p95_ms": 171.72
    },
    "OUT_OF_DOMAIN_REJECTION": {
      "count": 6,
      "mean_ms": 1.82,
      "p50_ms": 1.73,
      "p95_ms": 2.07
    },
    "MODALITY_GUARDRAIL": {
      "count": 2,
      "mean_ms": 28.71,
      "p50_ms": 28.71,
      "p95_ms": 31.19
    }
  },
  "sample_evaluations": [
    {
      "id": "vqa_001",
      "category": "single_image_vqa",
      "query": "Is there an airport runway visible in this scene?",
      "expected_task": "SINGLE_IMAGE_VQA",
      "predicted_task": "SINGLE_IMAGE_VQA",
      "expected_tool": "single_image_vqa",
      "predicted_tool": "single_image_vqa",
      "expected_validity": "VALID",
      "actual_validity": "VALID",
      "expected_error_code": null,
      "actual_error_code": null,
      "latency_ms": 515.26,
      "task_match": true,
      "tool_match": true,
      "overall_correct": true
    },

```

#### `reports/single_image.json`
```json
{
  "mission": "SatQuery AI Single-Image Remote Sensing Benchmark",
  "problem_statement": "ISRO / SAC \u2014 PS 26167",
  "evaluated_checkpoint": "SatQuery-RS-VLM-v1.0 (Adapted Checkpoint)",
  "runtime_device": "cpu",
  "timestamp": "2026-10-14T10:36:15Z",
  "datasets": {
    "rsvqa": {
      "dataset": "RSVQA (arXiv:2003.07333)",
      "split": "test",
      "total_samples": 36,
      "overall_accuracy_pct": 36.11,
      "breakdown_by_type": {
        "presence": {
          "total_questions": 11,
          "correct": 6,
          "accuracy_pct": 54.55
        },
        "count": {
          "total_questions": 9,
          "correct": 1,
          "accuracy_pct": 11.11
        },
        "comparison": {
          "total_questions": 8,
          "correct": 4,
          "accuracy_pct": 50.0
        },
        "area": {
          "total_questions": 8,
          "correct": 2,
          "accuracy_pct": 25.0
        }
      },
      "latency_sec": 41.69
    },
    "vrsbench_vqa": {
      "dataset": "VRSBench VQA (arXiv:2406.12384)",
      "split": "test",
      "total_samples": 30,
      "accuracy_pct": 76.67,
      "mean_lexical_f1_pct": 76.67,
      "latency_sec": 29.08
    },
    "vrsbench_captioning": {
      "dataset": "VRSBench Captioning (arXiv:2406.12384)",
      "split": "test",
      "total_samples": 15,
      "unigram_precision_pct": 3.9,
      "unigram_recall_pct": 7.78,
      "unigram_f1_pct": 5.2,
      "latency_sec": 23.63
    },
    "vrsbench_grounding": {
      "dataset": "VRSBench Grounding (arXiv:2406.12384)",
      "split": "test",
      "total_samples": 15,
      "acc_at_0_5_pct": 0.0,
      "mean_iou": 0.054,
      "hits_at_0_5": 0,
      "latency_sec": 0.13
    }
  },
  "summary": {
    "rsvqa_overall_accuracy": "36.11%",
    "rsvqa_presence_accuracy": "54.55%",
    "rsvqa_count_accuracy": "11.11%",
    "vrsbench_vqa_accuracy": "76.67%",
    "vrsbench_captioning_f1": "5.2%",
    "vrsbench_grounding_acc_0_5": "0.0%",
    "vrsbench_grounding_miou": "0.0540"
  }
}

```

#### `reports/change.json`
```json
{
  "benchmark": "CDVQA (arXiv:2112.06343)",
  "models": {
    "change_segmentation": "SatQueryChangeNet-v1.0 (Siamese U-Net, CDVQA trained)",
    "change_vqa": "SatQuery-RS-VLM-v1.0 (Adapted Checkpoint)"
  },
  "hardware": {
    "device": "cpu",
    "platform": "win32"
  },
  "dataset_statistics": {
    "test_pairs": 15,
    "question_types_evaluated": [
      "change_type",
      "existence",
      "quantity",
      "attribute"
    ]
  },
  "change_detection_mask_metrics": {
    "mean_iou_pct": 15.16,
    "mean_f1_dice": 0.2549,
    "precision": 0.1516,
    "recall": 0.8
  },
  "cdvqa_accuracy_metrics": {
    "overall_exact_accuracy_pct": 40.0,
    "overall_normalized_accuracy_pct": 40.0,
    "average_per_type_accuracy_pct": 25.0,
    "overall_lexical_f1": 0.4333,
    "per_question_type": {
      "change_type": {
        "count": 3,
        "exact_accuracy_pct": 0.0,
        "normalized_accuracy_pct": 0.0,
        "lexical_f1": 0.0
      },
      "existence": {
        "count": 6,
        "exact_accuracy_pct": 100.0,
        "normalized_accuracy_pct": 100.0,
        "lexical_f1": 1.0
      },
      "quantity": {
        "count": 3,
        "exact_accuracy_pct": 0.0,
        "normalized_accuracy_pct": 0.0,
        "lexical_f1": 0.1667
      },
      "attribute": {
        "count": 3,
        "exact_accuracy_pct": 0.0,
        "normalized_accuracy_pct": 0.0,
        "lexical_f1": 0.0
      }
    }
  },
  "runtime_seconds": 14.29,
  "sample_evaluations": [
    {
      "pair_id": "cdvqa_test_0000",
      "question": "What primary type of change occurred between the two dates?",
      "ground_truth": "urban development",
      "predicted": "date and time",
      "question_type": "change_type",
      "exact_match": false,
      "partial_match": false,
      "f1": 0.0,
      "mask_iou": 0.0,
      "mask_f1": 0.0
    },
    {
      "pair_id": "cdvqa_test_0001",
      "question": "Did the built-up area expand between T1 and T2?",
      "ground_truth": "yes",
      "predicted": "yes",
      "question_type": "existence",
      "exact_match": true,
      "partial_match": true,
      "f1": 1.0,
      "mask_iou": 0.1932367149758454,
      "mask_f1": 0.32388663967611336
    },
    {
      "pair_id": "cdvqa_test_0002",
      "question": "Has vegetation cover been cleared in the central sector?",
      "ground_truth": "yes",
      "predicted": "yes",
      "question_type": "existence",
      "exact_match": true,
      "partial_match": true,
      "f1": 1.0,
      "mask_iou": 0.1867745808689465,
      "mask_f1": 0.31475999550342865
    },
    {
      "pair_id": "cdvqa_test_0003",
      "question": "How many new structures were constructed in this timeframe?",
      "ground_truth": "0",
      "predicted": "1",
      "question_type": "quantity",
      "exact_match": false,
      "partial_match": false,
      "f1": 0.0,
      "mask_iou": 0.18660861065446305,
      "mask_f1": 0.31452428202343957
    },
    {
      "pair_id": "cdvqa_test_0004",
      "question": "What is the condition of the water body at T2 compared to T1?",
      "ground_truth": "receded shoreline",
      "predicted": "calm",
      "question_type": "attribute",
      "exact_match": false,
      "partial_match": false,
      "f1": 0.0,
      "mask_iou": 0.18665837073907826,
      "mask_f1": 0.3145949589903
    }
  ]
}

```

#### `reports/cross_modal.json`
```json
{
  "benchmark": "BigEarthNet.txt (arXiv:2603.29630)",
  "model": {
    "name": "SatQueryOpticalSARNet-v1.0 (Dual-Encoder Fusion Network, BigEarthNet trained)",
    "architecture": "SatQueryOpticalSARNet (Dual-Encoder Fusion Network)",
    "parameters": 849603
  },
  "hardware": {
    "device": "cpu",
    "platform": "win32"
  },
  "dataset_statistics": {
    "test_pairs_evaluated": 15,
    "sensor_modalities": "Sentinel-1 C-Band SAR + Sentinel-2 Multispectral"
  },
  "land_cover_extraction_metrics": {
    "overall_mean_iou_pct": 24.93,
    "overall_mean_f1_dice": 0.3049,
    "per_target_class": {
      "built_up": {
        "mean_iou_pct": 12.68,
        "dice_f1": 0.1533,
        "precision": 0.171,
        "recall": 0.148
      },
      "water": {
        "mean_iou_pct": 20.72,
        "dice_f1": 0.2927,
        "precision": 0.215,
        "recall": 0.5806
      },
      "vegetation": {
        "mean_iou_pct": 41.38,
        "dice_f1": 0.4689,
        "precision": 0.4307,
        "recall": 0.5293
      }
    }
  },
  "co_registration_guardrail_metrics": {
    "guardrail_verification_rate_pct": 100.0,
    "test_evaluations": [
      {
        "test_case": "True Co-Registered Pair (Mumbai Optical + RISAT-1 SAR)",
        "expected": "PASSED",
        "actual": "PASSED",
        "coherence_score": 0.88,
        "diagnostic_message": "Unverified Optical-SAR Co-Registration (Radiometric NMI: 0.0193, Gradient Coherence: 0.016)",
        "pass": true
      },
      {
        "test_case": "Disparate Regions (Bengaluru Optical + RISAT-1 SAR)",
        "expected": "REJECTED",
        "actual": "REJECTED",
        "coherence_score": 0.04,
        "diagnostic_message": "Cross-Modal Disparity Detected: Optical and SAR images appear to be from distinct geographic scenes (NMI: 0.0010, Gradient Corr: -0.013).",
        "pass": true
      }
    ]
  },
  "runtime_seconds": 9.26,
  "sample_evaluations": [
    {
      "patch_id": "S2A_MSIL2A_2018_test_0000",
      "clc_labels": [
        "Complex cultivation patterns",
        "Arable land",
        "Broad-leaved forest"
      ],
      "built_up_iou": 0.0,
      "water_iou": 0.0,
      "vegetation_iou": 0.3369
    },
    {
      "patch_id": "S2A_MSIL2A_2018_test_0001",
      "clc_labels": [
        "Urban fabric",
        "Industrial or commercial units",
        "Inland waters"
      ],
      "built_up_iou": 0.4378,
      "water_iou": 0.1482,
      "vegetation_iou": 0.0
    },
    {
      "patch_id": "S2A_MSIL2A_2018_test_0002",
      "clc_labels": [
        "Marine waters",
        "Beaches, dunes, sands",
        "Inland wetlands",
        "Natural grassland and sparsely vegetated areas"
      ],
      "built_up_iou": 0.0,
      "water_iou": 0.5883,
      "vegetation_iou": 0.9365
    },
    {
      "patch_id": "S2A_MSIL2A_2018_test_0003",
      "clc_labels": [
        "Coniferous forest",
        "Natural grassland and sparsely vegetated areas",
        "Transitional woodland, shrub"
      ],
      "built_up_iou": 0.0,
      "water_iou": 0.0,
      "vegetation_iou": 0.2711
    },
    {
      "patch_id": "S2A_MSIL2A_2018_test_0004",
      "clc_labels": [
        "Inland waters",
        "Pastures",
        "Land principally occupied by agriculture, with significant areas of natural vegetation"
      ],
      "built_up_iou": 0.0,
      "water_iou": 0.1452,
      "vegetation_iou": 0.5549
    }
  ]
}

```

#### `reports/latency.json`
```json
{
  "benchmark": "SatQuery-Latency-Profiling-v1.0",
  "timestamp": "2026-10-14T18:00:23+0530",
  "hardware": {
    "device": "Intel64 Family 6 Model 151 Stepping 2, GenuineIntel",
    "cuda_available": false,
    "torch_version": "2.13.0+cpu",
    "cpu_cores": 12,
    "os": "Windows 10"
  },
  "tools_profiled": 9,
  "benchmarks": {
    "single_image_vqa": {
      "iterations": 15,
      "mean_ms": 17.71,
      "p50_ms": 15.01,
      "p95_ms": 29.17,
      "min_ms": 13.32,
      "max_ms": 35.7,
      "std_ms": 5.89,
      "throughput_qps": 56.46
    },
    "single_image_captioning": {
      "iterations": 15,
      "mean_ms": 14.13,
      "p50_ms": 13.72,
      "p95_ms": 16.35,
      "min_ms": 13.3,
      "max_ms": 19.1,
      "std_ms": 1.41,
      "throughput_qps": 70.79
    },
    "single_image_grounding": {
      "iterations": 15,
      "mean_ms": 13.26,
      "p50_ms": 13.24,
      "p95_ms": 14.13,
      "min_ms": 12.55,
      "max_ms": 14.18,
      "std_ms": 0.51,
      "throughput_qps": 75.44
    },
    "land_cover_segmentation": {
      "iterations": 15,
      "mean_ms": 14.97,
      "p50_ms": 13.76,
      "p95_ms": 21.31,
      "min_ms": 12.25,
      "max_ms": 25.42,
      "std_ms": 3.39,
      "throughput_qps": 66.8
    },
    "bi_temporal_change": {
      "iterations": 15,
      "mean_ms": 50.55,
      "p50_ms": 51.76,
      "p95_ms": 58.05,
      "min_ms": 40.26,
      "max_ms": 58.35,
      "std_ms": 5.19,
      "throughput_qps": 19.78
    },
    "optical_sar_fusion": {
      "iterations": 15,
      "mean_ms": 7.16,
      "p50_ms": 7.07,
      "p95_ms": 8.59,
      "min_ms": 5.98,
      "max_ms": 8.6,
      "std_ms": 0.77,
      "throughput_qps": 139.75
    },
    "spatial_compatibility_verifier": {
      "iterations": 15,
      "mean_ms": 28.54,
      "p50_ms": 27.76,
      "p95_ms": 34.66,
      "min_ms": 25.95,
      "max_ms": 35.84,
      "std_ms": 2.8,
      "throughput_qps": 35.04
    },
    "large_raster_tiling_stitching": {
      "iterations": 10,
      "mean_ms": 236.86,
      "p50_ms": 229.29,
      "p95_ms": 274.93,
      "min_ms": 215.93,
      "max_ms": 291.6,
      "std_ms": 22.67,
      "throughput_qps": 4.22
    },
    "raster_ingestion_decode": {
      "iterations": 15,
      "mean_ms": 13.31,
      "p50_ms": 13.29,
      "p95_ms": 14.58,
      "min_ms": 11.86,
      "max_ms": 14.65,
      "std_ms": 0.85,
      "throughput_qps": 75.11
    }
  }
}

```

### 4.10 Unbacked Numbers Summary

The following numbers appear in `README.md`, `reports/final_matrix.md`, presentation decks, or UI labels but have **NO backing JSON file** in `reports/`:
1. **77.82% mIoU** (Cited for SatSegNet; actual backing file `ml_pipeline/evaluation_results.json` reports **80.22%**, while `training_metrics.json` reports **72.60%**).
2. **71.30% mIoU** (Cited for ChangeNet; actual backing file `reports/change.json` reports **15.16%** mask mIoU).
3. **75.80% mIoU** (Cited for OpticalSARNet; actual backing file `reports/cross_modal.json` reports **24.93%** overall mIoU).
4. **9.68 ms latency** (Cited in documentation; `reports/latency.json` contains no 9.68 ms entry; actual measured tool P50 latencies are 7.07 ms, 13.24 ms, 13.76 ms, etc.).

## 5. BENCHMARKS FROM THE PROBLEM STATEMENT

### 5.1 Public Benchmark Dataset Audit

| Benchmark Dataset | Reference Paper | Real Dataset Downloaded Locally? | Local Path & File Count | Evaluation Run on Real Split? | Evaluated Split & Metrics |
| :--- | :--- | :---: | :--- | :---: | :--- |
| **RSVQA** | Lobry et al. (arXiv:2003.07333) | **NO** (Real multi-gigabyte dataset not downloaded) | `ml_pipeline/dataset_cache/rsvqa/` (120 generated sample patches: 84 train, 18 val, 18 test) | **NO** (Evaluated on generated sample subset only) | **Overall Acc:** 36.11% (13/36)<br>• Presence: 54.55% (6/11)<br>• Comparison: 50.00% (4/8)<br>• Area: 25.00% (2/8)<br>• Count: 11.11% (1/9) |
| **VRSBench** | Ling et al. (arXiv:2406.12384) | **NO** (Real 50GB+ dataset not downloaded) | `ml_pipeline/dataset_cache/vrsbench/` (100 generated sample patches: 70 train, 15 val, 15 test) | **NO** (Evaluated on generated sample subset only) | **VQA Accuracy:** 76.67% (Mean Lexical F1: 76.67% on 30 questions)<br>**Captioning F1:** 5.2% (Unigram Prec: 3.9%, Recall: 7.78%)<br>**Grounding mIoU:** 0.0540 (Acc@0.5: 0.0%) |
| **CDVQA** | Yuan et al. (arXiv:2112.06343) | **NO** (Real bi-temporal dataset not downloaded) | `ml_pipeline/dataset_cache/cdvqa/` (100 generated sample pairs: 70 train, 15 val, 15 test) | **NO** (Evaluated on generated sample subset only) | **Overall Exact Match:** 40.0% (Overall Lexical F1: 0.4333)<br>• Existence: 100.0% (6/6)<br>• Quantity: 0.0% (Lexical F1: 0.1667)<br>• Change Type: 0.0%<br>• Attribute: 0.0%<br>**Mask mIoU:** 15.16% (Mean F1 Dice: 0.2549, Recall: 80.0%) |
| **BigEarthNet.txt** | Sumbul et al. (arXiv:2603.29630) | **NO** (Real 590k patch dataset not downloaded) | `ml_pipeline/dataset_cache/bigearthnet/` (100 generated sample pairs: 70 train, 15 val, 15 test) | **NO** (Evaluated on generated sample subset only) | **Modalities:** S1 SAR (VV/VH .npy) + S2 Optical (RGB .png)<br>**Patches:** 100 sample patches<br>**Model Trained on It:** SatQueryOpticalSARNet-v1.0 (trained 3 epochs on 70 train patches in `ml_pipeline/checkpoints/satquery_optical_sar_net/training_report.json`) |

## 6. ROUTING / CONTROLLER

### 6.1 Tool Registry Specifications (`backend/tool_registry.py`)
```python
"""
SatQuery AI — Typed Tool Registry (Pydantic v2)
Problem Statement Alignment: SIH26167 (ISRO / Space Applications Centre)

Defines strict schemas for all specialist remote sensing tools:
- Input specifications (min/max image count, allowed sensor modalities, file formats)
- Parameter specifications with type checks, numeric ranges, and allowed enums
- Output schemas (text answer, visual evidence, GeoJSON, measurements, confidence)
- Structured error representations with actionable remediation guidance
- Comprehensive execution trace schemas
"""
from enum import Enum
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field, field_validator, model_validator


# ─────────────────────────────────────────────────────────────────────────────
# Enums for Modality, Format, and Error Codes
# ─────────────────────────────────────────────────────────────────────────────

class SensorModality(str, Enum):
    OPTICAL = "OPTICAL"
    SAR = "SAR"
    MULTISPECTRAL = "MULTISPECTRAL"
    HYPERSPECTRAL = "HYPERSPECTRAL"
    UNKNOWN = "UNKNOWN"


class ImageFormat(str, Enum):
    GEOTIFF = "GEOTIFF"
    TIFF = "TIFF"
    PNG = "PNG"
    JPEG = "JPEG"


class StructuredErrorCode(str, Enum):
    INVALID_INPUT_COUNT = "INVALID_INPUT_COUNT"
    MODALITY_MISMATCH = "MODALITY_MISMATCH"
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
    NON_GEOSPATIAL_QUERY = "NON_GEOSPATIAL_QUERY"
    INVALID_PARAMETER = "INVALID_PARAMETER"
    MISSING_REQUIRED_PARAMETER = "MISSING_REQUIRED_PARAMETER"
    PARAMETER_OUT_OF_RANGE = "PARAMETER_OUT_OF_RANGE"
    SPATIAL_DISPARITY = "SPATIAL_DISPARITY"
    IDENTICAL_IMAGES = "IDENTICAL_IMAGES"
    EXECUTION_FAILURE = "EXECUTION_FAILURE"


# ─────────────────────────────────────────────────────────────────────────────
# Schema Definitions
# ─────────────────────────────────────────────────────────────────────────────

class InputRequirements(BaseModel):
    """Specification of input imagery requirements for a specialist tool."""
    min_images: int = Field(ge=0, le=10, description="Minimum number of input images required")
    max_images: int = Field(ge=0, le=10, description="Maximum number of input images accepted")
    allowed_modalities: List[SensorModality] = Field(
        default_factory=lambda: [SensorModality.OPTICAL, SensorModality.MULTISPECTRAL],
        description="List of permitted sensor modalities"
    )
    allowed_formats: List[ImageFormat] = Field(
        default_factory=lambda: [ImageFormat.GEOTIFF, ImageFormat.TIFF, ImageFormat.PNG, ImageFormat.JPEG],
        description="Supported raster file formats"
    )
    requires_georeferencing: bool = Field(
        default=False,
        description="Whether georeferenced raster metadata (CRS, Affine transform) is mandatory"
    )
    requires_cross_modal_pair: bool = Field(
        default=False,
        description="Whether tool strictly requires complementary modalities (e.g., 1 Optical + 1 SAR)"
    )


class ParameterSpec(BaseModel):
    """Specification and validation constraints for an individual tool parameter."""
    name: str
    param_type: str = Field(description="float | int | str | bool | list[str]")
    description: str
    default: Optional[Any] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[List[Any]] = None
    required: bool = False

    def validate_value(self, value: Any) -> Optional[str]:
        """Validate a parameter value against this specification. Returns error message or None."""
        if value is None:
            if self.required:
                return f"Missing required parameter '{self.name}'"
            return None

        # Type validation
        if self.param_type == "float":
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                return f"Parameter '{self.name}' must be a float, got {type(value).__name__}"
            if self.min_value is not None and value < self.min_value:
                return f"Parameter '{self.name}' value {value} is below minimum {self.min_value}"
            if self.max_value is not None and value > self.max_value:
                return f"Parameter '{self.name}' value {value} exceeds maximum {self.max_value}"

        elif self.param_type == "int":
            if not isinstance(value, int) or isinstance(value, bool):
                return f"Parameter '{self.name}' must be an integer, got {type(value).__name__}"
            if self.min_value is not None and value < self.min_value:
                return f"Parameter '{self.name}' value {value} is below minimum {int(self.min_value)}"
            if self.max_value is not None and value > self.max_value:
                return f"Parameter '{self.name}' value {value} exceeds maximum {int(self.max_value)}"

        elif self.param_type == "str":
            if not isinstance(value, str):
                return f"Parameter '{self.name}' must be a string, got {type(value).__name__}"
            if self.allowed_values and value not in self.allowed_values:
                return f"Parameter '{self.name}' value '{value}' is not permitted. Allowed: {self.allowed_values}"

        elif self.param_type == "bool":
            if not isinstance(value, bool):
                return f"Parameter '{self.name}' must be a boolean, got {type(value).__name__}"

        elif self.param_type == "list[str]":
            if not isinstance(value, (list, tuple)):
                return f"Parameter '{self.name}' must be a list of strings, got {type(value).__name__}"
            if not all(isinstance(v, str) for v in value):
                return f"Parameter '{self.name}' elements must all be strings"
            if self.allowed_values:
                for v in value:
                    if v not in self.allowed_values:
                        return f"Parameter '{self.name}' contains invalid element '{v}'. Allowed: {self.allowed_values}"

        return None


class OutputSchema(BaseModel):
    """Specification of outputs produced by a specialist tool."""
    has_text_answer: bool = True
    has_visual_evidence: bool = True
    has_geojson: bool = False
    has_mask: bool = False
    has_measurements: bool = False
    has_derived_confidence: bool = True


class ToolSpec(BaseModel):
    """Complete typed specification for a specialist remote sensing tool."""
    tool_id: str
    task_name: str
    version: str = "1.0.0"
    description: str
    inputs: InputRequirements
    parameters: Dict[str, ParameterSpec] = Field(default_factory=dict)
    output_schema: OutputSchema = Field(default_factory=OutputSchema)
    is_heuristic_fallback: bool = False
    provenance_label: str


class StructuredError(BaseModel):
    """Standardized structured error returned when validation or guardrails fail."""
    status: str = "error"
    error_code: StructuredErrorCode
    error_type: str
    message: str
    reason: str
    suggested_fix: str
    invalid_parameter: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)


class ToolCallRecord(BaseModel):
    """Record of an individual tool execution in the agentic trace."""
    tool_id: str
    task_name: str
    version: str
    status: str  # "SUCCESS" | "FAILED" | "REJECTED"
    latency_ms: float
    parameters_used: Dict[str, Any] = Field(default_factory=dict)
    summary: str = ""


class ExecutionTrace(BaseModel):
    """Full execution trace recording the 6-stage lifecycle of an agentic query."""
    trace_id: str
    timestamp: str
    query: str
    selected_task: str
    task_confidence: float
    input_checks: Dict[str, Any]
    tools_called: List[ToolCallRecord] = Field(default_factory=list)
    per_step_latency_ms: Dict[str, float] = Field(default_factory=dict)
    total_latency_ms: float
    outputs_summary: Dict[str, Any] = Field(default_factory=dict)
    confidence: float
    confidence_derivation: str
    valid_sequence: bool
    is_invalid_call: bool
    structured_error: Optional[StructuredError] = None


# ─────────────────────────────────────────────────────────────────────────────
# Master Tool Registry
# ─────────────────────────────────────────────────────────────────────────────

class ToolRegistry:
    """
    Central repository of typed specifications for all SatQuery AI tools.
    Provides validation of inputs, formats, parameters, and modalities.
    """

    def __init__(self):
        self._tools: Dict[str, ToolSpec] = {}
        self._register_default_tools()

    def register(self, spec: ToolSpec):
        """Register a new tool specification."""
        self._tools[spec.tool_id] = spec

    def get_tool(self, tool_id: str) -> Optional[ToolSpec]:
        """Retrieve tool specification by tool_id or task_name."""
        if tool_id in self._tools:
            return self._tools[tool_id]
        for tool in self._tools.values():
            if tool.task_name == tool_id:
                return tool
        return None

    def list_tools(self) -> List[ToolSpec]:
        """Return all registered tool specifications."""
        return list(self._tools.values())

    def _register_default_tools(self):
        # 1. Single Image VQA
        self.register(ToolSpec(
            tool_id="single_image_vqa",
            task_name="SINGLE_IMAGE_VQA",
            version="1.0.0",
            description="Visual Question Answering on single remote sensing images using RS-adapted VLM (RSVQA / VRSBench).",
            inputs=InputRequirements(
                min_images=1,
                max_images=1,
                allowed_modalities=[SensorModality.OPTICAL, SensorModality.MULTISPECTRAL],
                allowed_formats=[ImageFormat.GEOTIFF, ImageFormat.TIFF, ImageFormat.PNG, ImageFormat.JPEG],
                requires_georeferencing=False
            ),
            parameters={
                "question": ParameterSpec(
                    name="question",
                    param_type="str",
                    description="Natural language question about remote sensing imagery",
                    required=True
                ),
                "confidence_threshold": ParameterSpec(
                    name="confidence_threshold",
                    param_type="float",
                    description="Minimum confidence threshold for positive answers",
                    default=0.25,
                    min_value=0.0,
                    max_value=1.0
                ),
                "max_length": ParameterSpec(
                    name="max_length",
                    param_type="int",
                    description="Maximum token length for generated response",
                    default=128,
                    min_value=16,
                    max_value=256
                )
            },
            output_schema=OutputSchema(
                has_text_answer=True,
                has_visual_evidence=True,
                has_geojson=False,
                has_mask=False,
                has_measurements=False,
                has_derived_confidence=True
            ),
            is_heuristic_fallback=False,
            provenance_label="SatQuery-RS-VLM-v1.0 (BLIP RS-Adapted ViT-B/16)"
        ))

        # 2. Single Image Captioning
        self.register(ToolSpec(
            tool_id="single_image_captioning",
            task_name="SINGLE_IMAGE_CAPTIONING",
            version="1.0.0",
            description="Comprehensive scene description and semantic captioning for single remote sensing scenes.",
            inputs=InputRequirements(
                min_images=1,
                max_images=1,
                allowed_modalities=[SensorModality.OPTICAL, SensorModality.MULTISPECTRAL],
                allowed_formats=[ImageFormat.GEOTIFF, ImageFormat.TIFF, ImageFormat.PNG, ImageFormat.JPEG]
            ),
            parameters={
                "detail_level": ParameterSpec(
                    name="detail_level",
                    param_type="str",
                    description="Level of detail in generated scene caption",
                    default="standard",
                    allowed_values=["short", "standard", "comprehensive"]
                ),
                "max_length": ParameterSpec(
                    name="max_length",
                    param_type="int",
                    description="Maximum token length for caption",
                    default=180,
                    min_value=32,
                    max_value=512
                )
            },
            output_schema=OutputSchema(
                has_text_answer=True,
                has_visual_evidence=True,
                has_geojson=False,
                has_mask=False,
                has_measurements=True,
                has_derived_confidence=True
            ),
            is_heuristic_fallback=False,
            provenance_label="SatQuery-RS-VLM-v1.0 (VRSBench Captioning)"
        ))

        # 3. Single Image Grounding / Localization
        self.register(ToolSpec(
            tool_id="single_image_grounding",
            task_name="SINGLE_IMAGE_GROUNDING",
            version="1.0.0",
            description="Text-guided spatial localization and bounding box detection of geospatial targets.",
            inputs=InputRequirements(
                min_images=1,
                max_images=1,
                allowed_modalities=[SensorModality.OPTICAL, SensorModality.MULTISPECTRAL],
                allowed_formats=[ImageFormat.GEOTIFF, ImageFormat.TIFF, ImageFormat.PNG, ImageFormat.JPEG]
            ),
            parameters={
                "target_expression": ParameterSpec(
                    name="target_expression",
                    param_type="str",
                    description="Geospatial target or referring expression to ground",
                    required=True
                ),
                "iou_threshold": ParameterSpec(
                    name="iou_threshold",
                    param_type="float",
                    description="NMS IoU threshold for overlapping bounding boxes",
                    default=0.45,
                    min_value=0.1,
                    max_value=0.95
                ),
                "max_detections": ParameterSpec(
                    name="max_detections",
                    param_type="int",
                    description="Maximum number of objects to detect and ground",
                    default=50,
                    min_value=1,
                    max_value=200
                )
            },
            output_schema=OutputSchema(
                has_text_answer=True,
                has_visual_evidence=True,
                has_geojson=True,
                has_mask=False,
                has_measurements=True,
                has_derived_confidence=True
            ),
            is_heuristic_fallback=False,
            provenance_label="SatQuery Grounding Engine (VRSBench Grounding Head)"
        ))

        # 4. Land Cover Segmentation & Surface Quantification
        self.register(ToolSpec(
            tool_id="land_cover_segmentation",
            task_name="LAND_COVER_SEGMENTATION",
            version="1.0.0",
            description="Multi-class land-cover classification and surface area quantification in hectares and km².",
            inputs=InputRequirements(
                min_images=1,
                max_images=1,
                allowed_modalities=[SensorModality.OPTICAL, SensorModality.MULTISPECTRAL],
                allowed_formats=[ImageFormat.GEOTIFF, ImageFormat.TIFF, ImageFormat.PNG, ImageFormat.JPEG]
            ),
            parameters={
                "classes": ParameterSpec(
                    name="classes",
                    param_type="list[str]",
                    description="Target land-cover classes to quantify",
                    default=["water", "forest", "crop", "built_up", "bare_soil"],
                    allowed_values=["water", "forest", "crop", "built_up", "bare_soil", "cloud", "vegetation"]
                ),
                "calculate_area": ParameterSpec(
                    name="calculate_area",
                    param_type="bool",
                    description="Whether to calculate surface areas in km² and ha",
                    default=True
                ),
                "gsd_meters": ParameterSpec(
                    name="gsd_meters",
                    param_type="float",
                    description="Ground Sampling Distance in meters per pixel if unreferenced",
                    default=10.0,
                    min_value=0.1,
                    max_value=100.0
                )
            },
            output_schema=OutputSchema(
                has_text_answer=True,
                has_visual_evidence=True,
                has_geojson=True,
                has_mask=True,
                has_measurements=True,
                has_derived_confidence=True
            ),
            is_heuristic_fallback=False,
            provenance_label="SatQuery Land-Cover Engine"
        ))

        # 5. Bi-Temporal Change Analysis
        self.register(ToolSpec(
            tool_id="bi_temporal_change",
            task_name="CHANGE_ANALYSIS",
            version="1.0.0",
            description="Learned bi-temporal change detection and CDVQA difference question answering using SatQueryChangeNet.",
            inputs=InputRequirements(
                min_images=2,
                max_images=2,
                allowed_modalities=[SensorModality.OPTICAL, SensorModality.MULTISPECTRAL],
                allowed_formats=[ImageFormat.GEOTIFF, ImageFormat.TIFF, ImageFormat.PNG, ImageFormat.JPEG],
                requires_coregistration=True
            ),
            parameters={
                "change_threshold": ParameterSpec(
                    name="change_threshold",
                    param_type="float",
                    description="Confidence threshold for pixel difference segmentation",
                    default=0.35,
                    min_value=0.05,
                    max_value=0.95
                ),
                "min_contour_area_px": ParameterSpec(
                    name="min_contour_area_px",
                    param_type="int",
                    description="Minimum connected component pixel area to retain as a change polygon",
                    default=25,
                    min_value=5,
                    max_value=2000
                ),
                "quadrant_analysis": ParameterSpec(
                    name="quadrant_analysis",
                    param_type="bool",
                    description="Whether to segment changes into geographic/image quadrants",
                    default=True
                )
            },
            output_schema=OutputSchema(
                has_text_answer=True,
                has_visual_evidence=True,
                has_geojson=True,
                has_mask=True,
                has_measurements=True,
                has_derived_confidence=True
            ),
            is_heuristic_fallback=False,
            provenance_label="SatQueryChangeNet (Siamese Feature Diff U-Net, CDVQA trained)"
        ))

        # 6. Optical-SAR Joint Analysis
        self.register(ToolSpec(
            tool_id="optical_sar_fusion",
            task_name="CROSS_MODAL_EXTRACTION",
            version="1.0.0",
            description="Deep cross-modal joint reasoning combining Optical imagery with SAR microwave backscatter (SatQueryOpticalSARNet).",
            inputs=InputRequirements(
                min_images=2,
                max_images=2,
                allowed_modalities=[SensorModality.OPTICAL, SensorModality.SAR],
                allowed_formats=[ImageFormat.GEOTIFF, ImageFormat.TIFF, ImageFormat.PNG, ImageFormat.JPEG],
                requires_cross_modal_pair=True
            ),
            parameters={
                "target_classes": ParameterSpec(
                    name="target_classes",
                    param_type="list[str]",
                    description="Target classes for cross-modal extraction",
                    default=["water", "vegetation", "built_up"],
                    allowed_values=["water", "vegetation", "built_up"]
                ),
                "min_nmi": ParameterSpec(
                    name="min_nmi",
                    param_type="float",
                    description="Minimum Normalized Mutual Information for unreferenced co-registration",
                    default=0.010,
                    min_value=0.001,
                    max_value=0.1
                ),
                "confidence_threshold": ParameterSpec(
                    name="confidence_threshold",
                    param_type="float",
                    description="Threshold for multi-modal class probability activation",
                    default=0.30,
                    min_value=0.05,
                    max_value=0.95
                )
            },
            output_schema=OutputSchema(
                has_text_answer=True,
                has_visual_evidence=True,
                has_geojson=True,
                has_mask=True,
                has_measurements=True,
                has_derived_confidence=True
            ),
            is_heuristic_fallback=False,
            provenance_label="SatQueryOpticalSARNet (Dual-Encoder Fusion, BigEarthNet trained)"
        ))

        # 7. Spatial Compatibility Verifier
        self.register(ToolSpec(
            tool_id="spatial_compatibility_verifier",
            task_name="COMPATIBILITY_CHECK",
            version="1.0.0",
            description="Validates spatial co-registration across multi-temporal or cross-modal imagery pairs.",
            inputs=InputRequirements(
                min_images=2,
                max_images=2,
                allowed_modalities=[SensorModality.OPTICAL, SensorModality.SAR, SensorModality.MULTISPECTRAL],
                allowed_formats=[ImageFormat.GEOTIFF, ImageFormat.TIFF, ImageFormat.PNG, ImageFormat.JPEG]
            ),
            parameters={
                "verification_mode": ParameterSpec(
                    name="verification_mode",
                    param_type="str",
                    description="Verification algorithm to execute",
                    default="auto",
                    allowed_values=["auto", "optical_optical_sift", "optical_sar_nmi", "rasterio_crs_bounds"]
                ),
                "ransac_threshold": ParameterSpec(
                    name="ransac_threshold",
                    param_type="float",
                    description="RANSAC reprojection error threshold in pixels for SIFT",
                    default=5.0,
                    min_value=1.0,
                    max_value=20.0
                )
            },
            output_schema=OutputSchema(
                has_text_answer=True,
                has_visual_evidence=True,
                has_geojson=False,
                has_mask=False,
                has_measurements=True,
                has_derived_confidence=True
            ),
            is_heuristic_fallback=False,
            provenance_label="SatQuery Spatial Coherence Engine (Rasterio + SIFT/RANSAC + NMI)"
        ))

        # 8. Classical Heuristic Fallback (Strictly labeled)
        self.register(ToolSpec(
            tool_id="heuristic_fallback",
            task_name="FALLBACK_HEURISTIC",
            version="1.0.0",
            description="Classical rule-based computer vision routines (color thresholds, edge contours). Strictly designated as fallback.",
            inputs=InputRequirements(
                min_images=1,
                max_images=1,
                allowed_modalities=[SensorModality.OPTICAL, SensorModality.MULTISPECTRAL],
                allowed_formats=[ImageFormat.GEOTIFF, ImageFormat.TIFF, ImageFormat.PNG, ImageFormat.JPEG]
            ),
            parameters={
                "method": ParameterSpec(
                    name="method",
                    param_type="str",
                    description="Classical CV algorithm",
                    default="color_threshold",
                    allowed_values=["color_threshold", "edge_contour", "otsu"]
                )
            },
            output_schema=OutputSchema(
                has_text_answer=True,
                has_visual_evidence=True,
                has_geojson=False,
                has_mask=True,
                has_measurements=False,
                has_derived_confidence=True
            ),
            is_heuristic_fallback=True,
            provenance_label="SatQuery Classical CV Engine (fallback_heuristic)"
        ))

    # ─────────────────────────────────────────────────────────────────────────
    # Input & Parameter Validation Methods
    # ─────────────────────────────────────────────────────────────────────────

    def validate_inputs(
        self,
        tool_id: str,
        num_images: int,
        detected_modalities: List[str],
        detected_formats: List[str],
        benchmark_mode: bool = True
    ) -> Optional[StructuredError]:
        """Validate input count, modalities, and formats against tool requirements."""
        tool = self.get_tool(tool_id)
        if not tool:
            return StructuredError(
                error_code=StructuredErrorCode.INVALID_PARAMETER,
                error_type="UnknownToolError",
                message=f"Specialist tool '{tool_id}' is not registered.",
                reason="The controller attempted to dispatch an unregistered tool ID.",
                suggested_fix=f"Register tool '{tool_id}' or route to an existing registered tool."
            )

        reqs = tool.inputs

        # 1. Count validation
        if num_images < reqs.min_images or num_images > reqs.max_images:
            reason = (
                f"Tool '{tool.task_name}' requires between {reqs.min_images} and {reqs.max_images} images, "
                f"but {num_images} image(s) were supplied."
            )
            suggested = f"Provide exactly {reqs.min_images} image(s) for {tool.task_name}." if reqs.min_images == reqs.max_images else f"Provide {reqs.min_images} to {reqs.max_images} images."
            return StructuredError(
                error_code=StructuredErrorCode.INVALID_INPUT_COUNT,
                error_type="InputCountError",
                message=f"Invalid image count for {tool.task_name}: received {num_images}, expected {reqs.min_images}-{reqs.max_images}.",
                reason=reason,
                suggested_fix=suggested,
                details={"received_count": num_images, "expected_min": reqs.min_images, "expected_max": reqs.max_images}
            )

        # 2. Format validation (when benchmark mode is off, PNG/JPEG without georeferencing is rejected)
        if not benchmark_mode:
            for fmt in detected_formats:
                fmt_upper = fmt.upper()
                if fmt_upper in ("PNG", "JPEG", "JPG"):
                    return StructuredError(
                        error_code=StructuredErrorCode.UNSUPPORTED_FORMAT,
                        error_type="UnsupportedFormatError",
                        message=f"Standard raster format '{fmt}' is not permitted in production georeferenced mode.",
                        reason="Operational ISRO queries require calibrated 16-bit GeoTIFF imagery with CRS/transform metadata.",
                        suggested_fix="Upload a georeferenced GeoTIFF (.tif/.geotiff) or enable benchmark mode toggle for public PNG/JPEG datasets.",
                        details={"detected_format": fmt, "benchmark_mode": benchmark_mode}
                    )

        # 3. Cross-modal complementary pair check (Optical + SAR)
        if reqs.requires_cross_modal_pair:
            mod_set = {m.upper() for m in detected_modalities}
            has_optical = any(m in ("OPTICAL", "MULTISPECTRAL") for m in mod_set)
            has_sar = "SAR" in mod_set
            if not (has_optical and has_sar):
                return StructuredError(
                    error_code=StructuredErrorCode.MODALITY_MISMATCH,
                    error_type="CrossModalMismatchError",
                    message="Cross-modal analysis requires complementary Optical and SAR microwave imagery.",
                    reason=f"Detected modalities {detected_modalities} do not contain both Optical and SAR inputs.",
                    suggested_fix="Upload one Optical multispectral image and one Synthetic Aperture Radar (SAR) tile (e.g. RISAT-1 or Sentinel-1).",
                    details={"detected_modalities": detected_modalities}
                )

        # 4. Modality check against allowed modalities
        allowed_str = [m.value for m in reqs.allowed_modalities]
        for m in detected_modalities:
            if m.upper() not in allowed_str and m.upper() != "UNKNOWN":
                return StructuredError(
                    error_code=StructuredErrorCode.MODALITY_MISMATCH,
                    error_type="ModalityMismatchError",
                    message=f"Modality '{m}' is not supported by {tool.task_name}.",
                    reason=f"Tool '{tool.task_name}' accepts only {allowed_str}, but received '{m}'.",
                    suggested_fix=f"Provide imagery matching one of: {allowed_str}.",
                    details={"received_modality": m, "allowed_modalities": allowed_str}
                )

        return None

    def validate_parameters(
        self,
        tool_id: str,
        parameters: Dict[str, Any]
    ) -> Optional[StructuredError]:
        """Validate parameter dictionary against tool specification."""
        tool = self.get_tool(tool_id)
        if not tool:
            return StructuredError(
                error_code=StructuredErrorCode.INVALID_PARAMETER,
                error_type="UnknownToolError",
                message=f"Tool '{tool_id}' not found.",
                reason="Cannot validate parameters for an unknown tool.",
                suggested_fix="Verify the tool ID."
            )

        # Check for unknown parameters
        for p_name in parameters.keys():
            if p_name not in tool.parameters:
                return StructuredError(
                    error_code=StructuredErrorCode.INVALID_PARAMETER,
                    error_type="UnknownParameterError",
                    message=f"Unknown parameter '{p_name}' for tool '{tool.task_name}'.",
                    reason=f"Tool '{tool.task_name}' does not permit parameter '{p_name}'.",
                    suggested_fix=f"Permitted parameters: {list(tool.parameters.keys())}",
                    invalid_parameter=p_name,
                    details={"permitted_parameters": list(tool.parameters.keys())}
                )

        # Validate values of known parameters
        for p_name, p_spec in tool.parameters.items():
            val = parameters.get(p_name, p_spec.default)
            err = p_spec.validate_value(val)
            if err:
                code = StructuredErrorCode.PARAMETER_OUT_OF_RANGE if ("below minimum" in err or "exceeds maximum" in err) else StructuredErrorCode.INVALID_PARAMETER
                return StructuredError(
                    error_code=code,
                    error_type="ParameterValidationError",
                    message=err,
                    reason=f"Validation failed for parameter '{p_name}'.",
                    suggested_fix=f"Adjust '{p_name}' to meet constraints: {p_spec.description}",
                    invalid_parameter=p_name,
                    details={"parameter": p_name, "expected_type": p_spec.param_type}
                )

        return None


# Global singleton instance of ToolRegistry
tool_registry = ToolRegistry()

```

### 6.2 Tests Routing Cases Sample (`tests/routing_cases.jsonl`)

#### First 20 Lines
```jsonl
{"id": "vqa_001", "query": "Is there an airport runway visible in this scene?", "num_images": 1, "modalities": ["OPTICAL"], "formats": ["PNG"], "has_georeferencing": true, "benchmark_mode": true, "parameters": null, "expected_task": "SINGLE_IMAGE_VQA", "expected_tool": "single_image_vqa", "expected_validity": "VALID", "expected_error_code": null, "category": "single_image_vqa"}
{"id": "vqa_002", "query": "Are there any water bodies present in this image?", "num_images": 1, "modalities": ["OPTICAL"], "formats": ["PNG"], "has_georeferencing": true, "benchmark_mode": true, "parameters": null, "expected_task": "SINGLE_IMAGE_VQA", "expected_tool": "single_image_vqa", "expected_validity": "VALID", "expected_error_code": null, "category": "single_image_vqa"}
{"id": "vqa_003", "query": "How many storage tanks can be identified in this industrial area?", "num_images": 1, "modalities": ["OPTICAL"], "formats": ["PNG"], "has_georeferencing": true, "benchmark_mode": true, "parameters": null, "expected_task": "SINGLE_IMAGE_VQA", "expected_tool": "single_image_vqa", "expected_validity": "VALID", "expected_error_code": null, "category": "single_image_vqa"}
{"id": "vqa_004", "query": "Is there a bridge crossing the river?", "num_images": 1, "modalities": ["OPTICAL"], "formats": ["PNG"], "has_georeferencing": true, "benchmark_mode": true, "parameters": null, "expected_task": "SINGLE_IMAGE_VQA", "expected_tool": "single_image_vqa", "expected_validity": "VALID", "expected_error_code": null, "category": "single_image_vqa"}
{"id": "vqa_005", "query": "How many aircraft are stationed on the tarmac?", "num_images": 1, "modalities": ["OPTICAL"], "formats": ["PNG"], "has_georeferencing": true, "benchmark_mode": true, "parameters": null, "expected_task": "SINGLE_IMAGE_VQA", "expected_tool": "single_image_vqa", "expected_validity": "VALID", "expected_error_code": null, "category": "single_image_vqa"}
{"id": "vqa_006", "query": "Does the image contain residential buildings?", "num_images": 1, "modalities": ["OPTICAL"], "formats": ["PNG"], "has_georeferencing": true, "benchmark_mode": true, "parameters": null, "expected_task": "SINGLE_IMAGE_VQA", "expected_tool": "single_image_vqa", "expected_validity": "VALID", "expected_error_code": null, "category": "single_image_vqa"}
{"id": "vqa_007", "query": "What is the predominant land cover in the southern quadrant?", "num_images": 1, "modalities": ["OPTICAL"], "formats": ["PNG"], "has_georeferencing": true, "benchmark_mode": true, "parameters": null, "expected_task": "SINGLE_IMAGE_VQA", "expected_tool": "single_image_vqa", "expected_validity": "VALID", "expected_error_code": null, "category": "single_image_vqa"}
{"id": "vqa_008", "query": "Are there commercial vessels docked at the harbor?", "num_images": 1, "modalities": ["OPTICAL"], "formats": ["PNG"], "has_georeferencing": true, "benchmark_mode": true, "parameters": null, "expected_task": "SINGLE_IMAGE_VQA", "expected_tool": "single_image_vqa", "expected_validity": "VALID", "expected_error_code": null, "category": "single_image_vqa"}
{"id": "vqa_009", "query": "Is any railway infrastructure visible?", "num_images": 1, "modalities": ["OPTICAL"], "formats": ["PNG"], "has_georeferencing": true, "benchmark_mode": true, "parameters": null, "expected_task": "SINGLE_IMAGE_VQA", "expected_tool": "single_image_vqa", "expected_validity": "VALID", "expected_error_code": null, "category": "single_image_vqa"}
{"id": "vqa_010", "query": "Are there solar photovoltaic panels installed in this area?", "num_images": 1, "modalities": ["OPTICAL"], "formats": ["PNG"], "has_georeferencing": true, "benchmark_mode": true, "parameters": null, "expected_task": "SINGLE_IMAGE_VQA", "expected_tool": "single_image_vqa", "expected_validity": "VALID", "expected_error_code": null, "category": "single_image_vqa"}
{"id": "vqa_011", "query": "How many distinct building complexes can be seen?", "num_images": 1, "modalities": ["OPTICAL"], "formats": ["PNG"], "has_georeferencing": true, "benchmark_mode": true, "parameters": null, "expected_task": "SINGLE_IMAGE_VQA", "expected_tool": "single_image_vqa", "expected_validity": "VALID", "expected_error_code": null, "category": "single_image_vqa"}
{"id": "vqa_012", "query": "Is there active agricultural cultivation in these fields?", "num_images": 1, "modalities": ["OPTICAL"], "formats": ["PNG"], "has_georeferencing": true, "benchmark_mode": true, "parameters": null, "expected_task": "SINGLE_IMAGE_VQA", "expected_tool": "single_image_vqa", "expected_validity": "VALID", "expected_error_code": null, "category": "single_image_vqa"}
{"id": "vqa_013", "query": "Are there shipping containers stacked in the depot?", "num_images": 1, "modalities": ["OPTICAL"], "formats": ["PNG"], "has_georeferencing": true, "benchmark_mode": true, "parameters": null, "expected_task": "SINGLE_IMAGE_VQA", "expected_tool": "single_image_vqa", "expected_validity": "VALID", "expected_error_code": null, "category": "single_image_vqa"}
{"id": "vqa_014", "query": "Does this satellite tile show dense forest canopy?", "num_images": 1, "modalities": ["OPTICAL"], "formats": ["PNG"], "has_georeferencing": true, "benchmark_mode": true, "parameters": null, "expected_task": "SINGLE_IMAGE_VQA", "expected_tool": "single_image_vqa", "expected_validity": "VALID", "expected_error_code": null, "category": "single_image_vqa"}
{"id": "vqa_015", "query": "Is there an electrical transmission substation present?", "num_images": 1, "modalities": ["OPTICAL"], "formats": ["PNG"], "has_georeferencing": true, "benchmark_mode": true, "parameters": null, "expected_task": "SINGLE_IMAGE_VQA", "expected_tool": "single_image_vqa", "expected_validity": "VALID", "expected_error_code": null, "category": "single_image_vqa"}
{"id": "vqa_016", "query": "Are any vehicles visible on the primary highway?", "num_images": 1, "modalities": ["OPTICAL"], "formats": ["PNG"], "has_georeferencing": true, "benchmark_mode": true, "parameters": null, "expected_task": "SINGLE_IMAGE_VQA", "expected_tool": "single_image_vqa", "expected_validity": "VALID", "expected_error_code": null, "category": "single_image_vqa"}
{"id": "vqa_017", "query": "Is there a coastline or shoreline visible?", "num_images": 1, "modalities": ["OPTICAL"], "formats": ["PNG"], "has_georeferencing": true, "benchmark_mode": true, "parameters": null, "expected_task": "SINGLE_IMAGE_VQA", "expected_tool": "single_image_vqa", "expected_validity": "VALID", "expected_error_code": null, "category": "single_image_vqa"}
{"id": "vqa_018", "query": "How many industrial sheds are located in this sector?", "num_images": 1, "modalities": ["OPTICAL"], "formats": ["PNG"], "has_georeferencing": true, "benchmark_mode": true, "parameters": null, "expected_task": "SINGLE_IMAGE_VQA", "expected_tool": "single_image_vqa", "expected_validity": "VALID", "expected_error_code": null, "category": "single_image_vqa"}
{"id": "vqa_019", "query": "Is there evidence of terrain erosion or quarry excavation?", "num_images": 1, "modalities": ["OPTICAL"], "formats": ["PNG"], "has_georeferencing": true, "benchmark_mode": true, "parameters": null, "expected_task": "SINGLE_IMAGE_VQA", "expected_tool": "single_image_vqa", "expected_validity": "VALID", "expected_error_code": null, "category": "single_image_vqa"}
{"id": "vqa_020", "query": "Are there circular pivot irrigation fields in this scene?", "num_images": 1, "modalities": ["OPTICAL"], "formats": ["PNG"], "has_georeferencing": true, "benchmark_mode": true, "parameters": null, "expected_task": "SINGLE_IMAGE_VQA", "expected_tool": "single_image_vqa", "expected_validity": "VALID", "expected_error_code": null, "category": "single_image_vqa"}
```

#### 10 Randomly Sampled Lines (Seed 42)
```jsonl
{"id": "caption_019", "query": "What are the major environmental and surface characteristics shown?", "num_images": 1, "modalities": ["OPTICAL"], "formats": ["PNG"], "has_georeferencing": true, "benchmark_mode": true, "parameters": null, "expected_task": "SINGLE_IMAGE_CAPTIONING", "expected_tool": "single_image_captioning", "expected_validity": "VALID", "expected_error_code": null, "category": "single_image_captioning"}
{"id": "vqa_027", "query": "Is there a construction site with earthmoving activity?", "num_images": 1, "modalities": ["OPTICAL"], "formats": ["PNG"], "has_georeferencing": true, "benchmark_mode": true, "parameters": null, "expected_task": "SINGLE_IMAGE_VQA", "expected_tool": "single_image_vqa", "expected_validity": "VALID", "expected_error_code": null, "category": "single_image_vqa"}
{"id": "lulc_016", "query": "Calculate the coastal beach and sand dune surface area in ha.", "num_images": 1, "modalities": ["OPTICAL"], "formats": ["PNG"], "has_georeferencing": true, "benchmark_mode": true, "parameters": null, "expected_task": "LAND_COVER_SEGMENTATION", "expected_tool": "land_cover_segmentation", "expected_validity": "VALID", "expected_error_code": null, "category": "land_cover"}
{"id": "lulc_008", "query": "Compute the surface area of bare rocky soil in sq km.", "num_images": 1, "modalities": ["OPTICAL"], "formats": ["PNG"], "has_georeferencing": true, "benchmark_mode": true, "parameters": null, "expected_task": "LAND_COVER_SEGMENTATION", "expected_tool": "land_cover_segmentation", "expected_validity": "VALID", "expected_error_code": null, "category": "land_cover"}
{"id": "lulc_003", "query": "Measure the surface area of water bodies in square kilometers.", "num_images": 1, "modalities": ["OPTICAL"], "formats": ["PNG"], "has_georeferencing": true, "benchmark_mode": true, "parameters": null, "expected_task": "LAND_COVER_SEGMENTATION", "expected_tool": "land_cover_segmentation", "expected_validity": "VALID", "expected_error_code": null, "category": "land_cover"}
{"id": "ground_006", "query": "Show me where the railway junction and tracks are located.", "num_images": 1, "modalities": ["OPTICAL"], "formats": ["PNG"], "has_georeferencing": true, "benchmark_mode": true, "parameters": null, "expected_task": "SINGLE_IMAGE_GROUNDING", "expected_tool": "single_image_grounding", "expected_validity": "VALID", "expected_error_code": null, "category": "single_image_grounding"}
{"id": "caption_017", "query": "Tell me about the geographic features present in this image.", "num_images": 1, "modalities": ["OPTICAL"], "formats": ["PNG"], "has_georeferencing": true, "benchmark_mode": true, "parameters": null, "expected_task": "SINGLE_IMAGE_CAPTIONING", "expected_tool": "single_image_captioning", "expected_validity": "VALID", "expected_error_code": null, "category": "single_image_captioning"}
{"id": "adv_mod_001", "query": "Use SAR radar backscatter to analyze microwave dielectric roughness.", "num_images": 1, "modalities": ["OPTICAL"], "formats": ["PNG"], "has_georeferencing": true, "benchmark_mode": true, "parameters": null, "expected_task": "MODALITY_GUARDRAIL", "expected_tool": "optical_sar_fusion", "expected_validity": "INVALID", "expected_error_code": "MODALITY_MISMATCH", "category": "adversarial_modality"}
{"id": "caption_013", "query": "Summarize all prominent topographical and man-made structures.", "num_images": 1, "modalities": ["OPTICAL"], "formats": ["PNG"], "has_georeferencing": true, "benchmark_mode": true, "parameters": null, "expected_task": "SINGLE_IMAGE_CAPTIONING", "expected_tool": "single_image_captioning", "expected_validity": "VALID", "expected_error_code": null, "category": "single_image_captioning"}
{"id": "cross_004", "query": "Analyze microwave dielectric backscatter and optical reflectance for flood mapping.", "num_images": 2, "modalities": ["OPTICAL", "SAR"], "formats": ["PNG", "PNG"], "has_georeferencing": true, "benchmark_mode": true, "parameters": null, "expected_task": "CROSS_MODAL_EXTRACTION", "expected_tool": "optical_sar_fusion", "expected_validity": "VALID", "expected_error_code": null, "category": "cross_modal"}
```

#### Count of Lines Per Expected Task
```
Total cases evaluated: 167
Tasks breakdown:
  CHANGE_ANALYSIS: 32 cases
  SINGLE_IMAGE_VQA: 30 cases
  SINGLE_IMAGE_GROUNDING: 26 cases
  LAND_COVER_SEGMENTATION: 25 cases
  CROSS_MODAL_EXTRACTION: 24 cases
  SINGLE_IMAGE_CAPTIONING: 20 cases
  OUT_OF_DOMAIN_REJECTION: 8 cases
  MODALITY_GUARDRAIL: 2 cases

Expected Validity breakdown:
  VALID: 145 cases
  INVALID (Adversarial): 22 cases
```

### 6.3 Authorship & Provenance of Routing Cases
- **Who wrote these cases and how:** Hand-scripted by the developer in `scripts/generate_routing_benchmark.py` using domain templates and regex patterns matching the system's planned capabilities.
- **How many were adversarial / expected to fail:** Exactly **22 cases** were designed as adversarial / expected to be rejected:
  - `adversarial_ood`: 8 out-of-domain trivia/poem/code queries (expected error: `NON_GEOSPATIAL_QUERY`)
  - `adversarial_count`: 6 input count mismatches (e.g. 1 image provided for bi-temporal query; expected error: `INVALID_INPUT_COUNT`)
  - `adversarial_modality`: 4 modality mismatch queries (e.g. asking for SAR penetration with 2 optical images; expected error: `MODALITY_MISMATCH`)
  - `adversarial_format`: 1 unsupported format in production mode (expected error: `UNSUPPORTED_FORMAT`)
  - `adversarial_param`: 3 out-of-bounds parameter queries (expected error: `PARAMETER_OUT_OF_RANGE`)

### 6.4 Routing Confusion Matrix & Per-Task Accuracy
```json
{
  "metrics": {
    "task_classification_accuracy_pct": 91.02,
    "tool_selection_accuracy_pct": 91.02,
    "valid_sequence_rate_pct": 99.31,
    "invalid_call_rejection_rate_pct": 90.91,
    "structured_error_code_match_rate_pct": 86.36,
    "overall_accuracy_pct": 90.42
  },
  "per_category_metrics": {
    "single_image_vqa": {
      "total": 30,
      "accuracy_pct": 93.33,
      "task_acc_pct": 93.33,
      "tool_acc_pct": 93.33
    },
    "single_image_captioning": {
      "total": 20,
      "accuracy_pct": 70.0,
      "task_acc_pct": 70.0,
      "tool_acc_pct": 70.0
    },
    "single_image_grounding": {
      "total": 25,
      "accuracy_pct": 88.0,
      "task_acc_pct": 88.0,
      "tool_acc_pct": 88.0
    },
    "land_cover": {
      "total": 25,
      "accuracy_pct": 96.0,
      "task_acc_pct": 96.0,
      "tool_acc_pct": 96.0
    },
    "change": {
      "total": 25,
      "accuracy_pct": 100.0,
      "task_acc_pct": 100.0,
      "tool_acc_pct": 100.0
    },
    "cross_modal": {
      "total": 20,
      "accuracy_pct": 100.0,
      "task_acc_pct": 100.0,
      "tool_acc_pct": 100.0
    },
    "adversarial_ood": {
      "total": 8,
      "accuracy_pct": 75.0,
      "task_acc_pct": 75.0,
      "tool_acc_pct": 75.0
    },
    "adversarial_count": {
      "total": 6,
      "accuracy_pct": 83.33,
      "task_acc_pct": 83.33,
      "tool_acc_pct": 83.33
    },
    "adversarial_modality": {
      "total": 4,
      "accuracy_pct": 100.0,
      "task_acc_pct": 100.0,
      "tool_acc_pct": 100.0
    },
    "adversarial_format": {
      "total": 1,
      "accuracy_pct": 100.0,
      "task_acc_pct": 100.0,
      "tool_acc_pct": 100.0
    },
    "adversarial_param": {
      "total": 3,
      "accuracy_pct": 66.67,
      "task_acc_pct": 100.0,
      "tool_acc_pct": 100.0
    }
  }
}

```

### 6.5 Full Execution Trace JSON from a Real `/api/query` Call
```json
{
  "trace_id": "6e5c815b-151a-4b8b-a2c9-b0b8ebe931a9",
  "timestamp": "2026-10-14T19:55:30.976996",
  "query": "Is there a water body visible in this image?",
  "selected_task": "SINGLE_IMAGE_VQA",
  "task_confidence": 0.82,
  "input_checks": {
    "num_images": 1,
    "detected_modalities": [
      "OPTICAL"
    ],
    "detected_formats": [
      "PNG"
    ],
    "has_georeferencing": false,
    "benchmark_mode": true
  },
  "tools_called": [
    {
      "tool_id": "spatial_compatibility_verifier",
      "task_name": "COMPATIBILITY_CHECK",
      "version": "1.0.0",
      "status": "SUCCESS",
      "latency_ms": 123.93,
      "parameters_used": {
        "mode": "auto"
      },
      "summary": "Raster metadata & georeferencing inspection completed."
    },
    {
      "tool_id": "single_image_vqa",
      "task_name": "SINGLE_IMAGE_VQA",
      "version": "1.0.0",
      "status": "SUCCESS",
      "latency_ms": 477.07,
      "parameters_used": {},
      "summary": "Executed analysis pipeline using SatQuery CV Engine (OpenCV + ML Segmentation)."
    }
  ],
  "per_step_latency_ms": {
    "stage1_classification": 1.4,
    "stage2_input_verification": 123.93,
    "stage3_parameter_validation": 0.0,
    "stage4_tool_execution": 477.07
  },
  "total_latency_ms": 603.38,
  "outputs_summary": {
    "text_answer_length": 1173,
    "visual_evidence_count": 1,
    "geojson_features_count": 0,
    "measurements_count": 0,
    "detections_count": 0
  },
  "confidence": 0.82,
  "confidence_derivation": "Dynamically derived from model softmax probabilities, land-cover pixel coverage, and spatial validation agreement.",
  "valid_sequence": true,
  "is_invalid_call": false,
  "structured_error": null
}

```

## 7. BEHAVIOUR PROBES

All probes executed against live local backend daemon (`http://127.0.0.1:8000/api/query`). All base64 image payloads stripped.

### Probe a

- **Query:** `What is the capital of France?`

- **Input Image(s):** `['bengaluru_cartosat3.png']`

- **HTTP Status:** `200`

- **Response Time:** `666.78 ms`

- **Models Run (from trace):** `[]`

- **Fallback Heuristic Used:** `False`

- **Reported Confidence:** `0.04` (Code path: `backend/query_understanding.py:check_out_of_domain` line 386 — Guardrail refusal constant 0.04)

```json
{
  "status": "rejected",
  "answer": "\u26a0\ufe0f **Request Validation Error (NON_GEOSPATIAL_QUERY)**\n\nCannot answer from this remote sensing imagery: query seeks general trivia, programming, or mathematics unrelated to satellite imagery.\n\n\u2022 **Reason:** Query was classified as a non-geospatial question outside remote sensing observation capability.\n\u2022 **Action Required:** Ask questions pertaining to Earth observation, land cover, spatial infrastructure, or change over time.",
  "visual_evidence": [],
  "confidence": 0.04,
  "compatibility_status": "REJECTED_NON_GEOSPATIAL_QUERY",
  "spatial_coherence_score": 0.0,
  "execution_summary": {
    "selected_task": "OUT_OF_DOMAIN_REJECTION",
    "tool_used": "semantic_guardrail",
    "input_scope": "1 image(s)",
    "num_images_processed": 1,
    "agent_reasoning": "Validation failed at Stage 2/3: Query was classified as a non-geospatial question outside remote sensing observation capability."
  },
  "execution_trace": {
    "trace_id": "d9601def-c81e-490d-952b-a5a19de92b0c",
    "timestamp": "2026-10-14T19:55:29.264291",
    "query": "What is the capital of France?",
    "selected_task": "OUT_OF_DOMAIN_REJECTION",
    "task_confidence": 0.04,
    "input_checks": {
      "num_images": 1,
      "is_out_of_domain": true
    },
    "tools_called": [
      {
        "tool_id": "semantic_guardrail",
        "task_name": "OUT_OF_DOMAIN_REJECTION",
        "version": "1.0.0",
        "status": "REJECTED",
        "latency_ms": 16.2,
        "parameters_used": {},
        "summary": "Cannot answer from this remote sensing imagery: query seeks general trivia, programming, or mathematics unrelated to satellite imagery."
      }
    ],
    "per_step_latency_ms": {
      "input_validation": 16.2
    },
    "total_latency_ms": 16.2,
    "outputs_summary": {
      "status": "rejected",
      "error_code": "NON_GEOSPATIAL_QUERY",
      "visual_evidence_count": 0,
      "geojson_features": 0
    },
    "confidence": 0.04,
    "confidence_derivation": "Calibrated rejection uncertainty: 0.04 derived from guardrail constraint NON_GEOSPATIAL_QUERY.",
    "valid_sequence": false,
    "is_invalid_call": true,
    "structured_error": {
      "status": "error",
      "error_code": "NON_GEOSPATIAL_QUERY",
      "error_type": "OutOfDomainQueryError",
      "message": "Cannot answer from this remote sensing imagery: query seeks general trivia, programming, or mathematics unrelated to satellite imagery.",
      "reason": "Query was classified as a non-geospatial question outside remote sensing observation capability.",
      "suggested_fix": "Ask questions pertaining to Earth observation, land cover, spatial infrastructure, or change over time.",
      "invalid_parameter": null,
      "details": {
        "raw_query": "What is the capital of France?"
      }
    }
  },
  "structured_error": {
    "status": "error",
    "error_code": "NON_GEOSPATIAL_QUERY",
    "error_type": "OutOfDomainQueryError",
    "message": "Cannot answer from this remote sensing imagery: query seeks general trivia, programming, or mathematics unrelated to satellite imagery.",
    "reason": "Query was classified as a non-geospatial question outside remote sensing observation capability.",
    "suggested_fix": "Ask questions pertaining to Earth observation, land cover, spatial infrastructure, or change over time.",
    "invalid_parameter": null,
    "details": {
      "raw_query": "What is the capital of France?"
    }
  },
  "geo_metadata": null,
  "geojson_data": null
}
```

### Probe b

- **Query:** `Is there a nuclear submarine in this image?`

- **Input Image(s):** `['mumbai_cartosat2s_optical.png']`

- **HTTP Status:** `200`

- **Response Time:** `536.42 ms`

- **Models Run (from trace):** `[]`

- **Fallback Heuristic Used:** `False`

- **Reported Confidence:** `0.04` (Code path: `backend/query_understanding.py:check_out_of_domain` line 386 — Guardrail refusal constant 0.04)

```json
{
  "status": "rejected",
  "answer": "\u26a0\ufe0f **Request Validation Error (NON_GEOSPATIAL_QUERY)**\n\nCannot answer from this remote sensing imagery: subsurface marine vessels are not observable from optical remote sensing imagery.\n\n\u2022 **Reason:** Query was classified as a non-geospatial question outside remote sensing observation capability.\n\u2022 **Action Required:** Ask questions pertaining to Earth observation, land cover, spatial infrastructure, or change over time.",
  "visual_evidence": [],
  "confidence": 0.04,
  "compatibility_status": "REJECTED_NON_GEOSPATIAL_QUERY",
  "spatial_coherence_score": 0.0,
  "execution_summary": {
    "selected_task": "OUT_OF_DOMAIN_REJECTION",
    "tool_used": "semantic_guardrail",
    "input_scope": "1 image(s)",
    "num_images_processed": 1,
    "agent_reasoning": "Validation failed at Stage 2/3: Query was classified as a non-geospatial question outside remote sensing observation capability."
  },
  "execution_trace": {
    "trace_id": "8df25d4e-6846-4079-bdc0-ed1a44b9f652",
    "timestamp": "2026-10-14T19:55:29.813275",
    "query": "Is there a nuclear submarine in this image?",
    "selected_task": "OUT_OF_DOMAIN_REJECTION",
    "task_confidence": 0.04,
    "input_checks": {
      "num_images": 1,
      "is_out_of_domain": true
    },
    "tools_called": [
      {
        "tool_id": "semantic_guardrail",
        "task_name": "OUT_OF_DOMAIN_REJECTION",
        "version": "1.0.0",
        "status": "REJECTED",
        "latency_ms": 0.23,
        "parameters_used": {},
        "summary": "Cannot answer from this remote sensing imagery: subsurface marine vessels are not observable from optical remote sensing imagery."
      }
    ],
    "per_step_latency_ms": {
      "input_validation": 0.23
    },
    "total_latency_ms": 0.23,
    "outputs_summary": {
      "status": "rejected",
      "error_code": "NON_GEOSPATIAL_QUERY",
      "visual_evidence_count": 0,
      "geojson_features": 0
    },
    "confidence": 0.04,
    "confidence_derivation": "Calibrated rejection uncertainty: 0.04 derived from guardrail constraint NON_GEOSPATIAL_QUERY.",
    "valid_sequence": false,
    "is_invalid_call": true,
    "structured_error": {
      "status": "error",
      "error_code": "NON_GEOSPATIAL_QUERY",
      "error_type": "OutOfDomainQueryError",
      "message": "Cannot answer from this remote sensing imagery: subsurface marine vessels are not observable from optical remote sensing imagery.",
      "reason": "Query was classified as a non-geospatial question outside remote sensing observation capability.",
      "suggested_fix": "Ask questions pertaining to Earth observation, land cover, spatial infrastructure, or change over time.",
      "invalid_parameter": null,
      "details": {
        "raw_query": "Is there a nuclear submarine in this image?"
      }
    }
  },
  "structured_error": {
    "status": "error",
    "error_code": "NON_GEOSPATIAL_QUERY",
    "error_type": "OutOfDomainQueryError",
    "message": "Cannot answer from this remote sensing imagery: subsurface marine vessels are not observable from optical remote sensing imagery.",
    "reason": "Query was classified as a non-geospatial question outside remote sensing observation capability.",
    "suggested_fix": "Ask questions pertaining to Earth observation, land cover, spatial infrastructure, or change over time.",
    "invalid_parameter": null,
    "details": {
      "raw_query": "Is there a nuclear submarine in this image?"
    }
  },
  "geo_metadata": null,
  "geojson_data": null
}
```

### Probe c

- **Query:** `Is there a water body visible in this image?`

- **Input Image(s):** `['sambhar_lake_liss4.png']`

- **HTTP Status:** `200`

- **Response Time:** `3213.9 ms`

- **Models Run (from trace):** `[]`

- **Fallback Heuristic Used:** `False`

- **Reported Confidence:** `0.82` (Code path: `backend/model_interfaces.py:execute_water_body_detection` line 925 — Pixel detection density ratio min(0.95, 0.70 + 0.25 * coverage))

```json
{
  "status": "success",
  "answer": "\ud83d\udca7 **Water Body Delineation & Spatial Highlighting**\n\n### 1. Direct Grounding & Localization\n\u2022 **Visual Highlighting:** The water body has been delineated and **highlighted in luminous cyan-blue** with golden boundary vectors in the visual evidence canvas on the right.\n\u2022 **Surface Coverage:** Water bodies encompass approximately **~6.2%** (6.48 km\u00b2 / 648 hectares) of the observed scene.\n\n\n---\n\n### 2. Surface Distribution Balance\n  \u2022 **Built-Up Area**: **53.8%** (56.42 km\u00b2) / 5,642 ha\n  \u2022 **Cropland**: **19.1%** (20.02 km\u00b2) / 2,002 ha\n  \u2022 **Forest**: **18.6%** (19.49 km\u00b2) / 1,949 ha\n  \u2022 **Water Bodies**: **6.2%** (6.48 km\u00b2) / 648 ha\n  \u2022 **Cloud / Haze**: **2.3%** (2.44 km\u00b2) / 244 ha\n\n\n---\n\n### 3. Tactical GIS Interoperability\n\u2022 Vectorized boundary contours are ready for direct projection and export to **RFC 7946 WGS84 GeoJSON** (compatible with QGIS, ArcGIS, and ISRO Bhuvan).\n\n---\n\n**\u26a0 Limitations & Uncertainty**\n  \u2022 Requested target 'water_body' yielded null spatial distribution. The entity is either absent from the geographic bounding box, falls below the optical resolution limit, or lacks a dedicated neural extraction weight in the current configuration.",
  "visual_evidence": [
    {
      "image_base64": "<BASE64_IMAGE_STRIPPED_2725482_CHARS>",
      "description": "Delineated Water Bodies (~6.2%) & Terrestrial Landmass (~91.5%)"
    }
  ],
  "confidence": 0.82,
  "compatibility_status": "NOT_APPLICABLE",
  "spatial_coherence_score": 1.0,
  "execution_summary": {
    "selected_task": "SINGLE_IMAGE_VQA",
    "input_scope": "Single Image",
    "num_images_processed": 1,
    "agent_reasoning": "Query intent: WATER_BODY_ANALYSIS (confidence 90%). Targets: ['water_body']. Methods: ['land_cover_segmentation'].",
    "compatibility_status": "PASSED",
    "spatial_coherence_score": 1.0,
    "model_provenance": "SatQuery CV Engine (OpenCV + ML Segmentation)"
  },
  "execution_trace": {
    "trace_id": "6e5c815b-151a-4b8b-a2c9-b0b8ebe931a9",
    "timestamp": "2026-10-14T19:55:30.976996",
    "query": "Is there a water body visible in this image?",
    "selected_task": "SINGLE_IMAGE_VQA",
    "task_confidence": 0.82,
    "input_checks": {
      "num_images": 1,
      "detected_modalities": [
        "OPTICAL"
      ],
      "detected_formats": [
        "PNG"
      ],
      "has_georeferencing": false,
      "benchmark_mode": true
    },
    "tools_called": [
      {
        "tool_id": "spatial_compatibility_verifier",
        "task_name": "COMPATIBILITY_CHECK",
        "version": "1.0.0",
        "status": "SUCCESS",
        "latency_ms": 123.93,
        "parameters_used": {
          "mode": "auto"
        },
        "summary": "Raster metadata & georeferencing inspection completed."
      },
      {
        "tool_id": "single_image_vqa",
        "task_name": "SINGLE_IMAGE_VQA",
        "version": "1.0.0",
        "status": "SUCCESS",
        "latency_ms": 477.07,
        "parameters_used": {},
        "summary": "Executed analysis pipeline using SatQuery CV Engine (OpenCV + ML Segmentation)."
      }
    ],
    "per_step_latency_ms": {
      "stage1_classification": 1.4,
      "stage2_input_verification": 123.93,
      "stage3_parameter_validation": 0.0,
      "stage4_tool_execution": 477.07
    },
    "total_latency_ms": 603.38,
    "outputs_summary": {
      "text_answer_length": 1173,
      "visual_evidence_count": 1,
      "geojson_features_count": 0,
      "measurements_count": 0,
      "detections_count": 0
    },
    "confidence": 0.82,
    "confidence_derivation": "Dynamically derived from model softmax probabilities, land-cover pixel coverage, and spatial validation agreement.",
    "valid_sequence": true,
    "is_invalid_call": false,
    "structured_error": null
  },
  "geo_metadata": null,
  "geojson_data": null,
  "pair_comparison": null,
  "query_understanding": {
    "intent": "WATER_BODY_ANALYSIS",
    "sub_intents": [],
    "targets": [
      "water_body"
    ],
    "operation": "analyze",
    "spatial_constraint": null,
    "requires_count": false,
    "requires_area": false,
    "confidence": 0.9
  },
  "interpreted_task": "Water Body Detection",
  "detections": [],
  "detection_count": 0,
  "measurements": [],
  "limitations": [],
  "validation": {
    "passed": true,
    "issues": [],
    "warnings": [
      "Requested target 'water_body' yielded null spatial distribution. The entity is either absent from the geographic bounding box, falls below the optical resolution limit, or lacks a dedicated neural extraction weight in the current configuration."
    ]
  },
  "id": "3d2ff3aa-76b0-4b83-858c-4aebbebc3e43"
}
```

### Probe d

- **Query:** `Describe the land-cover and major objects visible in this image.`

- **Input Image(s):** `['mumbai_cartosat2s_optical.png']`

- **HTTP Status:** `200`

- **Response Time:** `7088.71 ms`

- **Models Run (from trace):** `[]`

- **Fallback Heuristic Used:** `False`

- **Reported Confidence:** `0.72` (Code path: `backend/model_interfaces.py:execute_scene_understanding` line 1445 — Land-cover Shannon entropy distribution confidence)

```json
{
  "status": "success",
  "answer": "\ud83d\udef0\ufe0f **Executive Scene & Land-Cover Assessment**\n\n### 1. Overview & Dominant Landscape\nThe scene shows a mix of **forest and vegetation** (70.0%), **agricultural land** (43.9%), **built-up area and settlements** (23.3%), **water bodies** (6.5%).\n\n---\n\n### 2. \ud83d\udcca Quantitative Land-Cover & Surface Breakdown\n\n| Surface Feature / Land Class | Surface Coverage | Estimated Area (km\u00b2) | Area (Hectares) | Status / Character |\n| :--- | :--- | :--- | :--- | :--- |\n| **Cropland** | **43.9%** | **46.00 km\u00b2** | **4,600 ha** | Delineated surface layer |\n| **Forest** | **26.1%** | **27.37 km\u00b2** | **2,737 ha** | Delineated surface layer |\n| **Built-Up Area** | **23.3%** | **24.47 km\u00b2** | **2,447 ha** | Delineated surface layer |\n| **Water Bodies** | **6.5%** | **6.80 km\u00b2** | **680 ha** | Delineated surface layer |\n| **Cloud / Haze** | **0.2%** | **0.22 km\u00b2** | **22 ha** | Delineated surface layer |\n\n---\n\n### 3. \ud83d\udccd Major Detected Objects & Infrastructure\n\u2022 Structural and linear features were integrated into the primary land-cover segmentation.\n\n---\n\n### 4. \ud83d\uddfa\ufe0f Tactical Evidence & GIS Readiness\n\u2022 All detected structures and surface boundaries are highlighted in the visual evidence canvas.\n\u2022 Full vector polygons are prepared for RFC 7946 GeoJSON export.\n\n---\n\n**\u26a0 Limitations & Uncertainty**\n  \u2022 Requested target class 'land' is abstract or unsupported by the active weights. System safely declined to hallucinate bounding boxes.\n  \u2022 Query requested spatial object localization, but no deterministic detections breached the confidence threshold. The scene may lack the specified structural signatures, or atmospheric interference compromised the feature extraction layer.\n  \u2022 Requested target 'land' yielded null spatial distribution. The entity is either absent from the geographic bounding box, falls below the optical resolution limit, or lacks a dedicated neural extraction weight in the current configuration.",
  "visual_evidence": [
    {
      "image_base64": "<BASE64_IMAGE_STRIPPED_2726274_CHARS>",
      "description": "AI Feature Salience & Focus Heatmap"
    }
  ],
  "confidence": 0.72,
  "compatibility_status": "NOT_APPLICABLE",
  "spatial_coherence_score": 1.0,
  "execution_summary": {
    "selected_task": "SINGLE_IMAGE_CAPTIONING",
    "input_scope": "Single Image",
    "num_images_processed": 1,
    "agent_reasoning": "Query intent: MULTI_TASK_QUERY (confidence 88%). Targets: ['land']. Methods: ['scene_description', 'object_detection', 'land_cover_segmentation'].",
    "compatibility_status": "PASSED",
    "spatial_coherence_score": 1.0,
    "model_provenance": "SatQuery CV Engine (OpenCV + ML Segmentation)"
  },
  "execution_trace": {
    "trace_id": "fddf96de-0ee5-490e-9961-7e0d94fec6a3",
    "timestamp": "2026-10-14T19:55:37.973229",
    "query": "Describe the land-cover and major objects visible in this image.",
    "selected_task": "SINGLE_IMAGE_CAPTIONING",
    "task_confidence": 0.72,
    "input_checks": {
      "num_images": 1,
      "detected_modalities": [
        "OPTICAL"
      ],
      "detected_formats": [
        "PNG"
      ],
      "has_georeferencing": false,
      "benchmark_mode": true
    },
    "tools_called": [
      {
        "tool_id": "spatial_compatibility_verifier",
        "task_name": "COMPATIBILITY_CHECK",
        "version": "1.0.0",
        "status": "SUCCESS",
        "latency_ms": 48.93,
        "parameters_used": {
          "mode": "auto"
        },
        "summary": "Raster metadata & georeferencing inspection completed."
      },
      {
        "tool_id": "single_image_captioning",
        "task_name": "SINGLE_IMAGE_CAPTIONING",
        "version": "1.0.0",
        "status": "SUCCESS",
        "latency_ms": 4318.94,
        "parameters_used": {},
        "summary": "Executed analysis pipeline using SatQuery CV Engine (OpenCV + ML Segmentation)."
      }
    ],
    "per_step_latency_ms": {
      "stage1_classification": 0.51,
      "stage2_input_verification": 48.93,
      "stage3_parameter_validation": 0.0,
      "stage4_tool_execution": 4318.94
    },
    "total_latency_ms": 4368.65,
    "outputs_summary": {
      "text_answer_length": 1909,
      "visual_evidence_count": 1,
      "geojson_features_count": 0,
      "measurements_count": 0,
      "detections_count": 0
    },
    "confidence": 0.72,
    "confidence_derivation": "Dynamically derived from model softmax probabilities, land-cover pixel coverage, and spatial validation agreement.",
    "valid_sequence": true,
    "is_invalid_call": false,
    "structured_error": null
  },
  "geo_metadata": null,
  "geojson_data": null,
  "pair_comparison": null,
  "query_understanding": {
    "intent": "MULTI_TASK_QUERY",
    "sub_intents": [
      "MULTI_TASK_QUERY",
      "SCENE_UNDERSTANDING",
      "LAND_COVER_CLASSIFICATION",
      "OBJECT_LOCALIZATION"
    ],
    "targets": [
      "land"
    ],
    "operation": "describe",
    "spatial_constraint": null,
    "requires_count": false,
    "requires_area": false,
    "confidence": 0.88
  },
  "interpreted_task": "Multi-Task Analysis",
  "detections": [],
  "detection_count": 0,
  "measurements": [],
  "limitations": [
    "Requested target class 'land' is abstract or unsupported by the active weights. System safely declined to hallucinate bounding boxes."
  ],
  "validation": {
    "passed": true,
    "issues": [],
    "warnings": [
      "Query requested spatial object localization, but no deterministic detections breached the confidence threshold. The scene may lack the specified structural signatures, or atmospheric interference compromised the feature extraction layer.",
      "Requested target 'land' yielded null spatial distribution. The entity is either absent from the geographic bounding box, falls below the optical resolution limit, or lacks a dedicated neural extraction weight in the current configuration."
    ]
  },
  "id": "a1868e11-2fc0-4a1d-b6d7-a8885fd86081"
}
```

### Probe e

- **Query:** `How many buildings are in this image?`

- **Input Image(s):** `['bengaluru_cartosat3.png']`

- **HTTP Status:** `200`

- **Response Time:** `2815.29 ms`

- **Models Run (from trace):** `[]`

- **Fallback Heuristic Used:** `False`

- **Reported Confidence:** `0.71` (Code path: `backend/object_detection.py:detect_objects` line 280 — Object detection count confidence formula)

```json
{
  "status": "success",
  "answer": "**DETECTION COUNT: Building**\n\n  Total detected: **0**\n\n\n\n---\n\n**\u26a0 Limitations & Uncertainty**\n  \u2022 Query requested spatial object localization, but no deterministic detections breached the confidence threshold. The scene may lack the specified structural signatures, or atmospheric interference compromised the feature extraction layer.\n  \u2022 Requested target 'building' yielded null spatial distribution. The entity is either absent from the geographic bounding box, falls below the optical resolution limit, or lacks a dedicated neural extraction weight in the current configuration.\n  \u2022 Count query detected but no count metric was produced.",
  "visual_evidence": [
    {
      "image_base64": "<BASE64_IMAGE_STRIPPED_3364842_CHARS>",
      "description": "AI Feature Salience & Focus Heatmap"
    }
  ],
  "confidence": 0.71,
  "compatibility_status": "NOT_APPLICABLE",
  "spatial_coherence_score": 1.0,
  "execution_summary": {
    "selected_task": "SINGLE_IMAGE_VQA",
    "input_scope": "Single Image",
    "num_images_processed": 1,
    "agent_reasoning": "Query intent: OBJECT_COUNTING (confidence 95%). Targets: ['building']. Methods: ['object_detection', 'count_objects'].",
    "compatibility_status": "PASSED",
    "spatial_coherence_score": 1.0,
    "model_provenance": "SatQuery CV Engine (OpenCV + ML Segmentation)"
  },
  "execution_trace": {
    "trace_id": "9ee387c0-4d02-4525-bf97-2e667af5e3aa",
    "timestamp": "2026-10-14T19:55:40.876526",
    "query": "How many buildings are in this image?",
    "selected_task": "SINGLE_IMAGE_VQA",
    "task_confidence": 0.71,
    "input_checks": {
      "num_images": 1,
      "detected_modalities": [
        "OPTICAL"
      ],
      "detected_formats": [
        "PNG"
      ],
      "has_georeferencing": false,
      "benchmark_mode": true
    },
    "tools_called": [
      {
        "tool_id": "spatial_compatibility_verifier",
        "task_name": "COMPATIBILITY_CHECK",
        "version": "1.0.0",
        "status": "SUCCESS",
        "latency_ms": 47.01,
        "parameters_used": {
          "mode": "auto"
        },
        "summary": "Raster metadata & georeferencing inspection completed."
      },
      {
        "tool_id": "single_image_vqa",
        "task_name": "SINGLE_IMAGE_VQA",
        "version": "1.0.0",
        "status": "SUCCESS",
        "latency_ms": 362.48,
        "parameters_used": {},
        "summary": "Executed analysis pipeline using SatQuery CV Engine (OpenCV + ML Segmentation)."
      }
    ],
    "per_step_latency_ms": {
      "stage1_classification": 0.41,
      "stage2_input_verification": 47.01,
      "stage3_parameter_validation": 0.0,
      "stage4_tool_execution": 362.48
    },
    "total_latency_ms": 410.05,
    "outputs_summary": {
      "text_answer_length": 642,
      "visual_evidence_count": 1,
      "geojson_features_count": 0,
      "measurements_count": 0,
      "detections_count": 0
    },
    "confidence": 0.71,
    "confidence_derivation": "Dynamically derived from model softmax probabilities, land-cover pixel coverage, and spatial validation agreement.",
    "valid_sequence": true,
    "is_invalid_call": false,
    "structured_error": null
  },
  "geo_metadata": null,
  "geojson_data": null,
  "pair_comparison": null,
  "query_understanding": {
    "intent": "OBJECT_COUNTING",
    "sub_intents": [],
    "targets": [
      "building"
    ],
    "operation": "count",
    "spatial_constraint": null,
    "requires_count": true,
    "requires_area": false,
    "confidence": 0.95
  },
  "interpreted_task": "Object Counting",
  "detections": [],
  "detection_count": 0,
  "measurements": [],
  "limitations": [],
  "validation": {
    "passed": true,
    "issues": [],
    "warnings": [
      "Query requested spatial object localization, but no deterministic detections breached the confidence threshold. The scene may lack the specified structural signatures, or atmospheric interference compromised the feature extraction layer.",
      "Requested target 'building' yielded null spatial distribution. The entity is either absent from the geographic bounding box, falls below the optical resolution limit, or lacks a dedicated neural extraction weight in the current configuration.",
      "Count query detected but no count metric was produced."
    ]
  },
  "id": "06c9af70-81c2-4b15-a2f2-43cf2a40eb6a"
}
```

### Probe f

- **Query:** `What changed between these two dates, and where did the change occur?`

- **Input Image(s):** `['uttarakhand_pre_event_t1.png', 'uttarakhand_post_event_t2.png']`

- **HTTP Status:** `200`

- **Response Time:** `5866.55 ms`

- **Models Run (from trace):** `[]`

- **Fallback Heuristic Used:** `False`

- **Reported Confidence:** `0.5605` (Code path: `backend/model_interfaces.py:execute_bitemporal_change` line 2145 — Mean softmax logit probability from SatQuery-RS-VLM)

```json
{
  "status": "success",
  "answer": "\ud83e\udd16 **VLM Change Analysis (SatQuery-RS-VLM-v1.0 (Adapted on BigEarthNet.txt, RSVQA, VRSBench)):** **water source**\n\n\ud83d\udd0d **Bi-Temporal Change Detection & Spatial Localization Assessment**\n\n### 1. Direct Answer: WHAT Changed Between the Two Dates\n\u2022 **Overall Change Extent:** Approximately **99.8%** (104.65 km\u00b2 / 10,465 hectares) of the scene experienced measurable surface transformation.\n\u2022 **Key Feature Shifts:**\n  \u2022 **Built-up Development:** Expanded by +20.6% (+21.64 km\u00b2)\n  \u2022 **Vegetation Cover:** Decreased by -14.3% (-14.95 km\u00b2)\n  \u2022 **Water Bodies:** Expanded by +4.4% (+4.64 km\u00b2)\n\n---\n\n### 2. Direct Answer: WHERE the Changes Occurred\n\u2022 **Change Cluster #1 (Central Core):** Urban Expansion / Structural Development \u2014 Spans **104.65 km\u00b2** (10,465 ha) | Position: `x:[0.00\u20131.00], y:[0.00\u20131.00]`\n\n---\n\n### \ud83d\udcca Quantitative Bi-Temporal Surface Transition Matrix\n\n| Surface Feature / Land Class | T1: Baseline Epoch | T2: Post-Event Epoch | Net Surface Shift (\u0394) | Estimated Area Shift (km\u00b2) | Transition Trajectory |\n| :--- | :--- | :--- | :--- | :--- | :--- |\n| \ud83c\udfd8\ufe0f **Urban Built-up & Infrastructure** | **46.5%** (48.79 km\u00b2) | **67.2%** (70.43 km\u00b2) | **+20.6%** | **+21.64 km\u00b2** (+2,164 ha) | \ud83d\udd3a Expansion |\n| \ud83c\udf33 **Vegetation & Tree Canopy** | **35.9%** (37.67 km\u00b2) | **21.7%** (22.72 km\u00b2) | **-14.3%** | **-14.95 km\u00b2** | \ud83d\udd3b Clearing |\n| \ud83d\udca7 **Water Bodies & Drainage** | **5.0%** (5.21 km\u00b2) | **9.4%** (9.85 km\u00b2) | **+4.4%** | **+4.64 km\u00b2** | \ud83d\udd3a Inundation |\n| \ud83e\udea8 **Bare Ground & Open Soil** | **0.0%** (0.01 km\u00b2) | **0.0%** (0.01 km\u00b2) | **+0.0%** | **+0.01 km\u00b2** | \u2696\ufe0f Stable |\n| \ud83d\udd0d **TOTAL DETECTED CHANGE EXTENT** | **Baseline (0.0%)** | **Active (99.8%)** | **99.8%** | **104.65 km\u00b2** (10,465 ha) | **SSIM Structural Deviation** |\n\n---\n\n### 3. Visual Evidence & Trace Verification\n\u2022 In the visual evidence canvas on the right, all change zones are demarcated with **red tactical bounding boxes** and sector IDs.\n\u2022 Use the **interactive before/after swipe slider** in the Trace panel to compare baseline and post-event satellite imagery side by side.",
  "visual_evidence": [
    {
      "image_base64": "<BASE64_IMAGE_STRIPPED_2682182_CHARS>",
      "description": "Bi-Temporal Change Map (99.8% Changed Area Demarcated in Red Bounding Boxes)"
    }
  ],
  "confidence": 0.5605,
  "compatibility_status": "PASSED",
  "spatial_coherence_score": 0.59,
  "execution_summary": {
    "selected_task": "CHANGE_ANALYSIS",
    "tool_used": "BiTemporalChangeAnalysis",
    "input_scope": "Multi-image (2 images)",
    "num_images_processed": 2,
    "agent_reasoning": "Multi-image query (2 images). Routed to CHANGE_ANALYSIS."
  },
  "execution_trace": {
    "trace_id": "2279bd19-9271-48eb-875b-0d8fa7422c17",
    "timestamp": "2026-10-14T19:55:46.276218",
    "query": "What changed between these two dates, and where did the change occur?",
    "selected_task": "CHANGE_ANALYSIS",
    "task_confidence": 0.5605,
    "input_checks": {
      "num_images": 2,
      "detected_modalities": [
        "OPTICAL",
        "OPTICAL"
      ],
      "detected_formats": [
        "PNG",
        "PNG"
      ],
      "has_georeferencing": false,
      "benchmark_mode": true
    },
    "tools_called": [
      {
        "tool_id": "spatial_compatibility_verifier",
        "task_name": "COMPATIBILITY_CHECK",
        "version": "1.0.0",
        "status": "SUCCESS",
        "latency_ms": 713.05,
        "parameters_used": {
          "mode": "auto"
        },
        "summary": "Spatial verification: PASSED (coherence: 0.59)"
      },
      {
        "tool_id": "bi_temporal_change",
        "task_name": "CHANGE_ANALYSIS",
        "version": "1.0.0",
        "status": "SUCCESS",
        "latency_ms": 2139.16,
        "parameters_used": {},
        "summary": "Executed BiTemporalChangeAnalysis generating answers, masks, and polygons."
      }
    ],
    "per_step_latency_ms": {
      "stage1_classification": 0.32,
      "stage2_input_verification": 62.8,
      "stage3_parameter_validation": 0.0,
      "stage4_tool_execution": 2852.22
    },
    "total_latency_ms": 2915.52,
    "outputs_summary": {
      "answer_length": 2045,
      "visual_evidence_count": 1,
      "geojson_features_count": 1
    },
    "confidence": 0.5605,
    "confidence_derivation": "Dynamically derived from spatial coherence score and model segmentation logit overlap.",
    "valid_sequence": true,
    "is_invalid_call": false,
    "structured_error": null
  },
  "geo_metadata": null,
  "geojson_data": {
    "type": "FeatureCollection",
    "metadata": {
      "mission": "SatQuery AI Bi-Temporal Change Detection & Spatial Localization",
      "coherence_score": 0.59,
      "change_percentage": 99.8,
      "total_change_km2": 104.65,
      "crs": "urn:ogc:def:crs:OGC:1.3:CRS84"
    },
    "features": [
      {
        "type": "Feature",
        "id": "change_polygon_1",
        "geometry": {
          "type": "Polygon",
          "coordinates": [
            [
              [
                79.71,
                30.43
              ],
              [
                79.75,
                30.43
              ],
              [
                79.75,
                30.39
              ],
              [
                79.71,
                30.39
              ],
              [
                79.71,
                30.43
              ]
            ]
          ]
        },
        "properties": {
          "change_id": 1,
          "quadrant": "Central Core",
          "change_type": "Urban Expansion / Structural Development",
          "area_km2": 104.65,
          "area_ha": 10465.0,
          "pixel_bbox": [
            0,
            0,
            1024,
            1024
          ]
        }
      }
    ]
  },
  "pair_comparison": {
    "type": "BI_TEMPORAL",
    "before_image": "<BASE64_IMAGE_STRIPPED_2673862_CHARS>",
    "after_image": "<BASE64_IMAGE_STRIPPED_2682182_CHARS>",
    "before_label": "T1: BASELINE OBSERVATION",
    "after_label": "T2: POST-EVENT / DETECTED CHANGES"
  },
  "query_understanding": {
    "intent": "CHANGE_DETECTION",
    "targets": [],
    "confidence": 0.65
  },
  "id": "7f1d4c09-600c-4161-908f-d639dd231d85"
}
```

### Probe g

- **Query:** `Has the built-up area increased, decreased, or remained unchanged?`

- **Input Image(s):** `['uttarakhand_pre_event_t1.png', 'uttarakhand_pre_event_t1.png']`

- **HTTP Status:** `200`

- **Response Time:** `860.98 ms`

- **Models Run (from trace):** `[]`

- **Fallback Heuristic Used:** `False`

- **Reported Confidence:** `0.04` (Code path: `backend/model_interfaces.py:check_spatial_compatibility` line 402 — Identical image hash match rejection)

```json
{
  "status": "rejected",
  "answer": "\u26a0\ufe0f **Identical Images Provided**\n\nIdentical images provided; cannot detect change. Bi-temporal change analysis requires two distinct observations captured at different temporal epochs (T1 baseline vs T2 post-event).",
  "visual_evidence": [],
  "confidence": 0.04,
  "compatibility_status": "REJECTED_IDENTICAL_IMAGES",
  "spatial_coherence_score": 1.0,
  "execution_summary": {
    "selected_task": "CHANGE_ANALYSIS",
    "tool_used": "BiTemporalChangeAnalysis",
    "input_scope": "Multi-image (2 images)",
    "num_images_processed": 2,
    "agent_reasoning": "Multi-image query (2 images). Routed to CHANGE_ANALYSIS."
  },
  "execution_trace": {
    "trace_id": "5f71a353-4f86-4739-a4cd-54de1ed0a5f1",
    "timestamp": "2026-10-14T19:55:49.715645",
    "query": "Has the built-up area increased, decreased, or remained unchanged?",
    "selected_task": "CHANGE_ANALYSIS",
    "task_confidence": 0.04,
    "input_checks": {
      "num_images": 2,
      "detected_modalities": [
        "OPTICAL",
        "OPTICAL"
      ],
      "detected_formats": [
        "PNG",
        "PNG"
      ],
      "has_georeferencing": false,
      "benchmark_mode": true
    },
    "tools_called": [
      {
        "tool_id": "spatial_compatibility_verifier",
        "task_name": "COMPATIBILITY_CHECK",
        "version": "1.0.0",
        "status": "REJECTED",
        "latency_ms": 1.35,
        "parameters_used": {
          "mode": "auto"
        },
        "summary": "Spatial verification: REJECTED_IDENTICAL_IMAGES (coherence: 1.0)"
      },
      {
        "tool_id": "bi_temporal_change",
        "task_name": "CHANGE_ANALYSIS",
        "version": "1.0.0",
        "status": "REJECTED",
        "latency_ms": 4.04,
        "parameters_used": {},
        "summary": "Executed BiTemporalChangeAnalysis generating answers, masks, and polygons."
      }
    ],
    "per_step_latency_ms": {
      "stage1_classification": 0.38,
      "stage2_input_verification": 74.25,
      "stage3_parameter_validation": 0.0,
      "stage4_tool_execution": 5.39
    },
    "total_latency_ms": 80.16,
    "outputs_summary": {
      "answer_length": 215,
      "visual_evidence_count": 0,
      "geojson_features_count": 0
    },
    "confidence": 0.04,
    "confidence_derivation": "Dynamically derived from spatial coherence score and model segmentation logit overlap.",
    "valid_sequence": false,
    "is_invalid_call": true,
    "structured_error": null
  },
  "geo_metadata": null,
  "geojson_data": null,
  "pair_comparison": null,
  "query_understanding": {
    "intent": "CHANGE_DETECTION",
    "targets": [
      "urban_area"
    ],
    "confidence": 0.9
  }
}
```

### Probe h

- **Query:** `What changed between these two dates?`

- **Input Image(s):** `['bengaluru_cartosat3.png', 'sambhar_lake_liss4.png']`

- **HTTP Status:** `200`

- **Response Time:** `2458.93 ms`

- **Models Run (from trace):** `[]`

- **Fallback Heuristic Used:** `False`

- **Reported Confidence:** `0.05` (Code path: `backend/model_interfaces.py:check_spatial_compatibility` line 516 — Weak keypoint match count (<15 inliers) score)

```json
{
  "status": "success",
  "answer": "\u26a0\ufe0f **Spatial Incompatibility: Images Show Different Geographic Locations**\n\nThe two uploaded images failed co-registration verification (Spatial coherence match score: **5%**).\n\n\u2022 **Co-registration Requirement:** Bi-temporal change detection requires two images capturing the exact same spatial bounding box.\n\u2022 **Action Required:** Please upload paired observations covering the same geospatial target at different acquisition epochs.",
  "visual_evidence": [
    {
      "image_base64": "<BASE64_IMAGE_STRIPPED_752174_CHARS>",
      "description": "Location Comparison (Mismatch)"
    }
  ],
  "confidence": 0.05,
  "compatibility_status": "FAILED",
  "spatial_coherence_score": 0.05,
  "execution_summary": {
    "selected_task": "CHANGE_ANALYSIS",
    "tool_used": "BiTemporalChangeAnalysis",
    "input_scope": "Multi-image (2 images)",
    "num_images_processed": 2,
    "agent_reasoning": "Multi-image query (2 images). Routed to CHANGE_ANALYSIS."
  },
  "execution_trace": {
    "trace_id": "57668e25-8875-4218-9cd8-71ba63b2b8b8",
    "timestamp": "2026-10-14T19:55:50.405812",
    "query": "What changed between these two dates?",
    "selected_task": "CHANGE_ANALYSIS",
    "task_confidence": 0.05,
    "input_checks": {
      "num_images": 2,
      "detected_modalities": [
        "OPTICAL",
        "OPTICAL"
      ],
      "detected_formats": [
        "PNG",
        "PNG"
      ],
      "has_georeferencing": false,
      "benchmark_mode": true
    },
    "tools_called": [
      {
        "tool_id": "spatial_compatibility_verifier",
        "task_name": "COMPATIBILITY_CHECK",
        "version": "1.0.0",
        "status": "SUCCESS",
        "latency_ms": 94.66,
        "parameters_used": {
          "mode": "auto"
        },
        "summary": "Spatial verification: FAILED (coherence: 0.05)"
      },
      {
        "tool_id": "bi_temporal_change",
        "task_name": "CHANGE_ANALYSIS",
        "version": "1.0.0",
        "status": "SUCCESS",
        "latency_ms": 283.99,
        "parameters_used": {},
        "summary": "Executed BiTemporalChangeAnalysis generating answers, masks, and polygons."
      }
    ],
    "per_step_latency_ms": {
      "stage1_classification": 0.24,
      "stage2_input_verification": 59.5,
      "stage3_parameter_validation": 0.0,
      "stage4_tool_execution": 378.65
    },
    "total_latency_ms": 438.5,
    "outputs_summary": {
      "answer_length": 434,
      "visual_evidence_count": 1,
      "geojson_features_count": 0
    },
    "confidence": 0.05,
    "confidence_derivation": "Dynamically derived from spatial coherence score and model segmentation logit overlap.",
    "valid_sequence": true,
    "is_invalid_call": false,
    "structured_error": null
  },
  "geo_metadata": null,
  "geojson_data": null,
  "pair_comparison": null,
  "query_understanding": {
    "intent": "CHANGE_DETECTION",
    "targets": [],
    "confidence": 0.65
  },
  "id": "cd2adc88-74a5-4ce0-9ba1-622e182f9165"
}
```

### Probe i

- **Query:** `Use the optical and SAR images together to identify built-up and water-covered regions.`

- **Input Image(s):** `['mumbai_cartosat2s_optical.png', 'risat1_cband_radar_sar.png']`

- **HTTP Status:** `200`

- **Response Time:** `366.64 ms`

- **Models Run (from trace):** `[]`

- **Fallback Heuristic Used:** `False`

- **Reported Confidence:** `0.04` (Code path: `backend/agent_controller.py:validate_inputs` line 165 — Modality check mismatch rejection)

```json
{
  "status": "rejected",
  "answer": "\u26a0\ufe0f **Request Validation Error (MODALITY_MISMATCH)**\n\nCross-modal analysis requires complementary Optical and SAR microwave imagery.\n\n\u2022 **Reason:** Detected modalities ['OPTICAL', 'OPTICAL'] do not contain both Optical and SAR inputs.\n\u2022 **Action Required:** Upload one Optical multispectral image and one Synthetic Aperture Radar (SAR) tile (e.g. RISAT-1 or Sentinel-1).",
  "visual_evidence": [],
  "confidence": 0.04,
  "compatibility_status": "REJECTED_MODALITY_MISMATCH",
  "spatial_coherence_score": 0.0,
  "execution_summary": {
    "selected_task": "CROSS_MODAL_EXTRACTION",
    "tool_used": "optical_sar_fusion",
    "input_scope": "2 image(s)",
    "num_images_processed": 2,
    "agent_reasoning": "Validation failed at Stage 2/3: Detected modalities ['OPTICAL', 'OPTICAL'] do not contain both Optical and SAR inputs."
  },
  "execution_trace": {
    "trace_id": "9937ba7e-8515-4008-9d1c-60bf62329c58",
    "timestamp": "2026-10-14T19:55:52.545004",
    "query": "Use the optical and SAR images together to identify built-up and water-covered regions.",
    "selected_task": "CROSS_MODAL_EXTRACTION",
    "task_confidence": 0.04,
    "input_checks": {
      "num_images": 2,
      "detected_modalities": [
        "OPTICAL",
        "OPTICAL"
      ],
      "detected_formats": [
        "PNG",
        "PNG"
      ],
      "has_georeferencing": false,
      "benchmark_mode": true
    },
    "tools_called": [
      {
        "tool_id": "optical_sar_fusion",
        "task_name": "CROSS_MODAL_EXTRACTION",
        "version": "1.0.0",
        "status": "REJECTED",
        "latency_ms": 59.93,
        "parameters_used": {},
        "summary": "Cross-modal analysis requires complementary Optical and SAR microwave imagery."
      }
    ],
    "per_step_latency_ms": {
      "input_validation": 59.93
    },
    "total_latency_ms": 59.93,
    "outputs_summary": {
      "status": "rejected",
      "error_code": "MODALITY_MISMATCH",
      "visual_evidence_count": 0,
      "geojson_features": 0
    },
    "confidence": 0.04,
    "confidence_derivation": "Calibrated rejection uncertainty: 0.04 derived from guardrail constraint MODALITY_MISMATCH.",
    "valid_sequence": false,
    "is_invalid_call": true,
    "structured_error": {
      "status": "error",
      "error_code": "MODALITY_MISMATCH",
      "error_type": "CrossModalMismatchError",
      "message": "Cross-modal analysis requires complementary Optical and SAR microwave imagery.",
      "reason": "Detected modalities ['OPTICAL', 'OPTICAL'] do not contain both Optical and SAR inputs.",
      "suggested_fix": "Upload one Optical multispectral image and one Synthetic Aperture Radar (SAR) tile (e.g. RISAT-1 or Sentinel-1).",
      "invalid_parameter": null,
      "details": {
        "detected_modalities": [
          "OPTICAL",
          "OPTICAL"
        ]
      }
    }
  },
  "structured_error": {
    "status": "error",
    "error_code": "MODALITY_MISMATCH",
    "error_type": "CrossModalMismatchError",
    "message": "Cross-modal analysis requires complementary Optical and SAR microwave imagery.",
    "reason": "Detected modalities ['OPTICAL', 'OPTICAL'] do not contain both Optical and SAR inputs.",
    "suggested_fix": "Upload one Optical multispectral image and one Synthetic Aperture Radar (SAR) tile (e.g. RISAT-1 or Sentinel-1).",
    "invalid_parameter": null,
    "details": {
      "detected_modalities": [
        "OPTICAL",
        "OPTICAL"
      ]
    }
  },
  "geo_metadata": null,
  "geojson_data": null
}
```

### Probe j

- **Query:** `Use the optical and SAR images together to identify built-up and water-covered regions.`

- **Input Image(s):** `['bengaluru_cartosat3.png', 'risat1_cband_radar_sar.png']`

- **HTTP Status:** `200`

- **Response Time:** `268.65 ms`

- **Models Run (from trace):** `[]`

- **Fallback Heuristic Used:** `False`

- **Reported Confidence:** `0.04` (Code path: `backend/agent_controller.py:validate_inputs` line 165 — Modality check mismatch rejection)

```json
{
  "status": "rejected",
  "answer": "\u26a0\ufe0f **Request Validation Error (MODALITY_MISMATCH)**\n\nCross-modal analysis requires complementary Optical and SAR microwave imagery.\n\n\u2022 **Reason:** Detected modalities ['OPTICAL', 'OPTICAL'] do not contain both Optical and SAR inputs.\n\u2022 **Action Required:** Upload one Optical multispectral image and one Synthetic Aperture Radar (SAR) tile (e.g. RISAT-1 or Sentinel-1).",
  "visual_evidence": [],
  "confidence": 0.04,
  "compatibility_status": "REJECTED_MODALITY_MISMATCH",
  "spatial_coherence_score": 0.0,
  "execution_summary": {
    "selected_task": "CROSS_MODAL_EXTRACTION",
    "tool_used": "optical_sar_fusion",
    "input_scope": "2 image(s)",
    "num_images_processed": 2,
    "agent_reasoning": "Validation failed at Stage 2/3: Detected modalities ['OPTICAL', 'OPTICAL'] do not contain both Optical and SAR inputs."
  },
  "execution_trace": {
    "trace_id": "e51f71b0-0a87-4202-975e-65f035cfbfe8",
    "timestamp": "2026-10-14T19:55:52.814836",
    "query": "Use the optical and SAR images together to identify built-up and water-covered regions.",
    "selected_task": "CROSS_MODAL_EXTRACTION",
    "task_confidence": 0.04,
    "input_checks": {
      "num_images": 2,
      "detected_modalities": [
        "OPTICAL",
        "OPTICAL"
      ],
      "detected_formats": [
        "PNG",
        "PNG"
      ],
      "has_georeferencing": false,
      "benchmark_mode": true
    },
    "tools_called": [
      {
        "tool_id": "optical_sar_fusion",
        "task_name": "CROSS_MODAL_EXTRACTION",
        "version": "1.0.0",
        "status": "REJECTED",
        "latency_ms": 67.61,
        "parameters_used": {},
        "summary": "Cross-modal analysis requires complementary Optical and SAR microwave imagery."
      }
    ],
    "per_step_latency_ms": {
      "input_validation": 67.61
    },
    "total_latency_ms": 67.61,
    "outputs_summary": {
      "status": "rejected",
      "error_code": "MODALITY_MISMATCH",
      "visual_evidence_count": 0,
      "geojson_features": 0
    },
    "confidence": 0.04,
    "confidence_derivation": "Calibrated rejection uncertainty: 0.04 derived from guardrail constraint MODALITY_MISMATCH.",
    "valid_sequence": false,
    "is_invalid_call": true,
    "structured_error": {
      "status": "error",
      "error_code": "MODALITY_MISMATCH",
      "error_type": "CrossModalMismatchError",
      "message": "Cross-modal analysis requires complementary Optical and SAR microwave imagery.",
      "reason": "Detected modalities ['OPTICAL', 'OPTICAL'] do not contain both Optical and SAR inputs.",
      "suggested_fix": "Upload one Optical multispectral image and one Synthetic Aperture Radar (SAR) tile (e.g. RISAT-1 or Sentinel-1).",
      "invalid_parameter": null,
      "details": {
        "detected_modalities": [
          "OPTICAL",
          "OPTICAL"
        ]
      }
    }
  },
  "structured_error": {
    "status": "error",
    "error_code": "MODALITY_MISMATCH",
    "error_type": "CrossModalMismatchError",
    "message": "Cross-modal analysis requires complementary Optical and SAR microwave imagery.",
    "reason": "Detected modalities ['OPTICAL', 'OPTICAL'] do not contain both Optical and SAR inputs.",
    "suggested_fix": "Upload one Optical multispectral image and one Synthetic Aperture Radar (SAR) tile (e.g. RISAT-1 or Sentinel-1).",
    "invalid_parameter": null,
    "details": {
      "detected_modalities": [
        "OPTICAL",
        "OPTICAL"
      ]
    }
  },
  "geo_metadata": null,
  "geojson_data": null
}
```

### Probe k

- **Query:** `What changed between these two dates?`

- **Input Image(s):** `['bengaluru_cartosat3.png']`

- **HTTP Status:** `200`

- **Response Time:** `276.55 ms`

- **Models Run (from trace):** `[]`

- **Fallback Heuristic Used:** `False`

- **Reported Confidence:** `0.04` (Code path: `backend/agent_controller.py:validate_inputs` line 145 — Input image count mismatch rejection)

```json
{
  "status": "rejected",
  "answer": "\u26a0\ufe0f **Request Validation Error (INVALID_INPUT_COUNT)**\n\nBi-temporal change detection requires exactly 2 images (T1 pre-event and T2 post-event), but only 1 image was supplied.\n\n\u2022 **Reason:** Change detection computes pixel-level and semantic surface transitions between two distinct temporal observations.\n\u2022 **Action Required:** Please upload two co-registered satellite images: a baseline (T1) and a subsequent observation (T2).",
  "visual_evidence": [],
  "confidence": 0.04,
  "compatibility_status": "REJECTED_INVALID_INPUT_COUNT",
  "spatial_coherence_score": 0.0,
  "execution_summary": {
    "selected_task": "CHANGE_ANALYSIS",
    "tool_used": "bi_temporal_change",
    "input_scope": "1 image(s)",
    "num_images_processed": 1,
    "agent_reasoning": "Validation failed at Stage 2/3: Change detection computes pixel-level and semantic surface transitions between two distinct temporal observations."
  },
  "execution_trace": {
    "trace_id": "aa541d17-978e-48a3-83ad-ce08f59d9f89",
    "timestamp": "2026-10-14T19:55:53.092044",
    "query": "What changed between these two dates?",
    "selected_task": "CHANGE_ANALYSIS",
    "task_confidence": 0.04,
    "input_checks": {
      "num_images": 1,
      "detected_modalities": [
        "OPTICAL"
      ],
      "detected_formats": [
        "PNG"
      ],
      "has_georeferencing": false,
      "benchmark_mode": true
    },
    "tools_called": [
      {
        "tool_id": "bi_temporal_change",
        "task_name": "CHANGE_ANALYSIS",
        "version": "1.0.0",
        "status": "REJECTED",
        "latency_ms": 46.2,
        "parameters_used": {},
        "summary": "Bi-temporal change detection requires exactly 2 images (T1 pre-event and T2 post-event), but only 1 image was supplied."
      }
    ],
    "per_step_latency_ms": {
      "input_validation": 46.2
    },
    "total_latency_ms": 46.2,
    "outputs_summary": {
      "status": "rejected",
      "error_code": "INVALID_INPUT_COUNT",
      "visual_evidence_count": 0,
      "geojson_features": 0
    },
    "confidence": 0.04,
    "confidence_derivation": "Calibrated rejection uncertainty: 0.04 derived from guardrail constraint INVALID_INPUT_COUNT.",
    "valid_sequence": false,
    "is_invalid_call": true,
    "structured_error": {
      "status": "error",
      "error_code": "INVALID_INPUT_COUNT",
      "error_type": "InputCountError",
      "message": "Bi-temporal change detection requires exactly 2 images (T1 pre-event and T2 post-event), but only 1 image was supplied.",
      "reason": "Change detection computes pixel-level and semantic surface transitions between two distinct temporal observations.",
      "suggested_fix": "Please upload two co-registered satellite images: a baseline (T1) and a subsequent observation (T2).",
      "invalid_parameter": null,
      "details": {
        "received_images": 1,
        "required_images": 2
      }
    }
  },
  "structured_error": {
    "status": "error",
    "error_code": "INVALID_INPUT_COUNT",
    "error_type": "InputCountError",
    "message": "Bi-temporal change detection requires exactly 2 images (T1 pre-event and T2 post-event), but only 1 image was supplied.",
    "reason": "Change detection computes pixel-level and semantic surface transitions between two distinct temporal observations.",
    "suggested_fix": "Please upload two co-registered satellite images: a baseline (T1) and a subsequent observation (T2).",
    "invalid_parameter": null,
    "details": {
      "received_images": 1,
      "required_images": 2
    }
  },
  "geo_metadata": null,
  "geojson_data": null
}
```

### Probe l

- **Query:** `Is there a water body visible?`

- **Input Image(s):** `['uttarakhand_pre_event_t1.png', 'uttarakhand_post_event_t2.png']`

- **HTTP Status:** `200`

- **Response Time:** `5245.08 ms`

- **Models Run (from trace):** `[]`

- **Fallback Heuristic Used:** `False`

- **Reported Confidence:** `0.5605` (Code path: `backend/model_interfaces.py:execute_bitemporal_change` line 2145 — Two images routed to bi-temporal change analysis)

```json
{
  "status": "success",
  "answer": "\ud83e\udd16 **VLM Change Analysis (SatQuery-RS-VLM-v1.0 (Adapted on BigEarthNet.txt, RSVQA, VRSBench)):** **yes**\n\n\ud83d\udd0d **Bi-Temporal Change Detection & Spatial Localization Assessment**\n\n### 1. Direct Answer: WHAT Changed Between the Two Dates\n\u2022 **Overall Change Extent:** Approximately **99.8%** (104.65 km\u00b2 / 10,465 hectares) of the scene experienced measurable surface transformation.\n\u2022 **Key Feature Shifts:**\n  \u2022 **Built-up Development:** Expanded by +20.6% (+21.64 km\u00b2)\n  \u2022 **Vegetation Cover:** Decreased by -14.3% (-14.95 km\u00b2)\n  \u2022 **Water Bodies:** Expanded by +4.4% (+4.64 km\u00b2)\n\n---\n\n### 2. Direct Answer: WHERE the Changes Occurred\n\u2022 **Change Cluster #1 (Central Core):** Urban Expansion / Structural Development \u2014 Spans **104.65 km\u00b2** (10,465 ha) | Position: `x:[0.00\u20131.00], y:[0.00\u20131.00]`\n\n---\n\n### \ud83d\udcca Quantitative Bi-Temporal Surface Transition Matrix\n\n| Surface Feature / Land Class | T1: Baseline Epoch | T2: Post-Event Epoch | Net Surface Shift (\u0394) | Estimated Area Shift (km\u00b2) | Transition Trajectory |\n| :--- | :--- | :--- | :--- | :--- | :--- |\n| \ud83c\udfd8\ufe0f **Urban Built-up & Infrastructure** | **46.5%** (48.79 km\u00b2) | **67.2%** (70.43 km\u00b2) | **+20.6%** | **+21.64 km\u00b2** (+2,164 ha) | \ud83d\udd3a Expansion |\n| \ud83c\udf33 **Vegetation & Tree Canopy** | **35.9%** (37.67 km\u00b2) | **21.7%** (22.72 km\u00b2) | **-14.3%** | **-14.95 km\u00b2** | \ud83d\udd3b Clearing |\n| \ud83d\udca7 **Water Bodies & Drainage** | **5.0%** (5.21 km\u00b2) | **9.4%** (9.85 km\u00b2) | **+4.4%** | **+4.64 km\u00b2** | \ud83d\udd3a Inundation |\n| \ud83e\udea8 **Bare Ground & Open Soil** | **0.0%** (0.01 km\u00b2) | **0.0%** (0.01 km\u00b2) | **+0.0%** | **+0.01 km\u00b2** | \u2696\ufe0f Stable |\n| \ud83d\udd0d **TOTAL DETECTED CHANGE EXTENT** | **Baseline (0.0%)** | **Active (99.8%)** | **99.8%** | **104.65 km\u00b2** (10,465 ha) | **SSIM Structural Deviation** |\n\n---\n\n### 3. Visual Evidence & Trace Verification\n\u2022 In the visual evidence canvas on the right, all change zones are demarcated with **red tactical bounding boxes** and sector IDs.\n\u2022 Use the **interactive before/after swipe slider** in the Trace panel to compare baseline and post-event satellite imagery side by side.",
  "visual_evidence": [
    {
      "image_base64": "<BASE64_IMAGE_STRIPPED_2682182_CHARS>",
      "description": "Bi-Temporal Change Map (99.8% Changed Area Demarcated in Red Bounding Boxes)"
    }
  ],
  "confidence": 0.5605,
  "compatibility_status": "PASSED",
  "spatial_coherence_score": 0.59,
  "execution_summary": {
    "selected_task": "CHANGE_ANALYSIS",
    "tool_used": "BiTemporalChangeAnalysis",
    "input_scope": "Multi-image (2 images)",
    "num_images_processed": 2,
    "agent_reasoning": "Multi-image query (2 images). Routed to CHANGE_ANALYSIS."
  },
  "execution_trace": {
    "trace_id": "2fe917de-4b23-4dc0-9ff7-bfc354a4ec07",
    "timestamp": "2026-10-14T19:55:55.861672",
    "query": "Is there a water body visible?",
    "selected_task": "CHANGE_ANALYSIS",
    "task_confidence": 0.5605,
    "input_checks": {
      "num_images": 2,
      "detected_modalities": [
        "OPTICAL",
        "OPTICAL"
      ],
      "detected_formats": [
        "PNG",
        "PNG"
      ],
      "has_georeferencing": false,
      "benchmark_mode": true
    },
    "tools_called": [
      {
        "tool_id": "spatial_compatibility_verifier",
        "task_name": "COMPATIBILITY_CHECK",
        "version": "1.0.0",
        "status": "SUCCESS",
        "latency_ms": 578.36,
        "parameters_used": {
          "mode": "auto"
        },
        "summary": "Spatial verification: PASSED (coherence: 0.59)"
      },
      {
        "tool_id": "bi_temporal_change",
        "task_name": "CHANGE_ANALYSIS",
        "version": "1.0.0",
        "status": "SUCCESS",
        "latency_ms": 1735.08,
        "parameters_used": {},
        "summary": "Executed BiTemporalChangeAnalysis generating answers, masks, and polygons."
      }
    ],
    "per_step_latency_ms": {
      "stage1_classification": 0.32,
      "stage2_input_verification": 73.45,
      "stage3_parameter_validation": 0.0,
      "stage4_tool_execution": 2313.44
    },
    "total_latency_ms": 2387.35,
    "outputs_summary": {
      "answer_length": 2036,
      "visual_evidence_count": 1,
      "geojson_features_count": 1
    },
    "confidence": 0.5605,
    "confidence_derivation": "Dynamically derived from spatial coherence score and model segmentation logit overlap.",
    "valid_sequence": true,
    "is_invalid_call": false,
    "structured_error": null
  },
  "geo_metadata": null,
  "geojson_data": {
    "type": "FeatureCollection",
    "metadata": {
      "mission": "SatQuery AI Bi-Temporal Change Detection & Spatial Localization",
      "coherence_score": 0.59,
      "change_percentage": 99.8,
      "total_change_km2": 104.65,
      "crs": "urn:ogc:def:crs:OGC:1.3:CRS84"
    },
    "features": [
      {
        "type": "Feature",
        "id": "change_polygon_1",
        "geometry": {
          "type": "Polygon",
          "coordinates": [
            [
              [
                79.71,
                30.43
              ],
              [
                79.75,
                30.43
              ],
              [
                79.75,
                30.39
              ],
              [
                79.71,
                30.39
              ],
              [
                79.71,
                30.43
              ]
            ]
          ]
        },
        "properties": {
          "change_id": 1,
          "quadrant": "Central Core",
          "change_type": "Urban Expansion / Structural Development",
          "area_km2": 104.65,
          "area_ha": 10465.0,
          "pixel_bbox": [
            0,
            0,
            1024,
            1024
          ]
        }
      }
    ]
  },
  "pair_comparison": {
    "type": "BI_TEMPORAL",
    "before_image": "<BASE64_IMAGE_STRIPPED_2673862_CHARS>",
    "after_image": "<BASE64_IMAGE_STRIPPED_2682182_CHARS>",
    "before_label": "T1: BASELINE OBSERVATION",
    "after_label": "T2: POST-EVENT / DETECTED CHANGES"
  },
  "query_understanding": {
    "intent": "COMPARISON",
    "targets": [
      "water_body"
    ],
    "confidence": 0.6
  },
  "id": "319e5146-44c6-4296-b8f0-08bfa3cf630e"
}
```

### Probe m1_geotiff

- **Query:** `Is there a water body visible in this scene?`

- **Input Image(s):** `['2k_earth_normal_map.tif']`

- **HTTP Status:** `200`

- **Response Time:** `2754.68 ms`

- **Models Run (from trace):** `[]`

- **Fallback Heuristic Used:** `False`

- **Reported Confidence:** `0.82` (Code path: `backend/model_interfaces.py:execute_water_body_detection` line 925 — Processed 2k_earth_normal_map.tif as visual TIFF)

```json
{
  "status": "success",
  "answer": "\ud83d\udca7 **Water Body Delineation & Spatial Highlighting**\n\n### 1. Direct Grounding & Localization\n\u2022 **Visual Highlighting:** The water body has been delineated and **highlighted in luminous cyan-blue** with golden boundary vectors in the visual evidence canvas on the right.\n\u2022 **Surface Coverage:** Water bodies encompass approximately **~100.0%** (52.43 km\u00b2 / 5,243 hectares) of the observed scene.\n\n\n---\n\n### 2. Surface Distribution Balance\n  \u2022 **Water Bodies**: **100.0%** (52.43 km\u00b2) / 5,243 ha\n\n\n---\n\n### 3. Tactical GIS Interoperability\n\u2022 Vectorized boundary contours are ready for direct projection and export to **RFC 7946 WGS84 GeoJSON** (compatible with QGIS, ArcGIS, and ISRO Bhuvan).\n\n---\n\n**\u26a0 Limitations & Uncertainty**\n  \u2022 Requested target 'water_body' yielded null spatial distribution. The entity is either absent from the geographic bounding box, falls below the optical resolution limit, or lacks a dedicated neural extraction weight in the current configuration.",
  "visual_evidence": [
    {
      "image_base64": "<BASE64_IMAGE_STRIPPED_126598_CHARS>",
      "description": "Delineated Water Bodies (~100.0%) & Terrestrial Landmass (~0.0%)"
    }
  ],
  "confidence": 0.82,
  "compatibility_status": "NOT_APPLICABLE",
  "spatial_coherence_score": 1.0,
  "execution_summary": {
    "selected_task": "SINGLE_IMAGE_VQA",
    "input_scope": "Single Image",
    "num_images_processed": 1,
    "agent_reasoning": "Query intent: WATER_BODY_ANALYSIS (confidence 90%). Targets: ['water_body']. Methods: ['land_cover_segmentation'].",
    "compatibility_status": "PASSED",
    "spatial_coherence_score": 1.0,
    "model_provenance": "SatQuery CV Engine (OpenCV + ML Segmentation)"
  },
  "execution_trace": {
    "trace_id": "3b219cc7-457d-4a82-8b86-adf4fbf11852",
    "timestamp": "2026-10-14T19:55:59.107251",
    "query": "Is there a water body visible in this scene?",
    "selected_task": "SINGLE_IMAGE_VQA",
    "task_confidence": 0.82,
    "input_checks": {
      "num_images": 1,
      "detected_modalities": [
        "OPTICAL"
      ],
      "detected_formats": [
        "GEOTIFF"
      ],
      "has_georeferencing": false,
      "benchmark_mode": true
    },
    "tools_called": [
      {
        "tool_id": "spatial_compatibility_verifier",
        "task_name": "COMPATIBILITY_CHECK",
        "version": "1.0.0",
        "status": "SUCCESS",
        "latency_ms": 269.33,
        "parameters_used": {
          "mode": "auto"
        },
        "summary": "Raster metadata & georeferencing inspection completed."
      },
      {
        "tool_id": "single_image_vqa",
        "task_name": "SINGLE_IMAGE_VQA",
        "version": "1.0.0",
        "status": "SUCCESS",
        "latency_ms": 218.25,
        "parameters_used": {},
        "summary": "Executed analysis pipeline using SatQuery CV Engine (OpenCV + ML Segmentation)."
      }
    ],
    "per_step_latency_ms": {
      "stage1_classification": 0.22,
      "stage2_input_verification": 269.33,
      "stage3_parameter_validation": 0.0,
      "stage4_tool_execution": 218.25
    },
    "total_latency_ms": 487.88,
    "outputs_summary": {
      "text_answer_length": 976,
      "visual_evidence_count": 1,
      "geojson_features_count": 0,
      "measurements_count": 0,
      "detections_count": 0
    },
    "confidence": 0.82,
    "confidence_derivation": "Dynamically derived from model softmax probabilities, land-cover pixel coverage, and spatial validation agreement.",
    "valid_sequence": true,
    "is_invalid_call": false,
    "structured_error": null
  },
  "geo_metadata": null,
  "geojson_data": null,
  "pair_comparison": null,
  "query_understanding": {
    "intent": "WATER_BODY_ANALYSIS",
    "sub_intents": [],
    "targets": [
      "water_body"
    ],
    "operation": "describe",
    "spatial_constraint": null,
    "requires_count": false,
    "requires_area": false,
    "confidence": 0.9
  },
  "interpreted_task": "Water Body Detection",
  "detections": [],
  "detection_count": 0,
  "measurements": [],
  "limitations": [],
  "validation": {
    "passed": true,
    "issues": [],
    "warnings": [
      "Requested target 'water_body' yielded null spatial distribution. The entity is either absent from the geographic bounding box, falls below the optical resolution limit, or lacks a dedicated neural extraction weight in the current configuration."
    ]
  },
  "id": "f7e2cd41-419a-4fbc-b8b0-e42c2fc5785c"
}
```

### Probe m2_corrupt

- **Query:** `Is there a water body visible in this image?`

- **Input Image(s):** `['corrupt_image.png']`

- **HTTP Status:** `200`

- **Response Time:** `694.0 ms`

- **Models Run (from trace):** `[]`

- **Fallback Heuristic Used:** `False`

- **Reported Confidence:** `None` (Code path: `backend/main.py:process_query` line 285 — PIL/rasterio decode failure returning HTTP 200 error payload)

```json
{
  "status": "error",
  "message": "Unsupported or corrupted image format. Please ensure the file is a valid satellite or visual image (GeoTIFF, TIFF, JP2, PNG, JPEG, WebP, AVIF, BMP)."
}
```

### Probe m3_three_images

- **Query:** `Analyze these three images.`

- **Input Image(s):** `['bengaluru_cartosat3.png', 'mumbai_cartosat2s_optical.png', 'sambhar_lake_liss4.png']`

- **HTTP Status:** `200`

- **Response Time:** `616.41 ms`

- **Models Run (from trace):** `[]`

- **Fallback Heuristic Used:** `False`

- **Reported Confidence:** `0.04` (Code path: `backend/agent_controller.py:validate_inputs` line 145 — 3 images exceeding max allowed images rejection)

```json
{
  "status": "rejected",
  "answer": "\u26a0\ufe0f **Request Validation Error (INVALID_INPUT_COUNT)**\n\nInvalid image count for CHANGE_ANALYSIS: received 3, expected 2-2.\n\n\u2022 **Reason:** Tool 'CHANGE_ANALYSIS' requires between 2 and 2 images, but 3 image(s) were supplied.\n\u2022 **Action Required:** Provide exactly 2 image(s) for CHANGE_ANALYSIS.",
  "visual_evidence": [],
  "confidence": 0.04,
  "compatibility_status": "REJECTED_INVALID_INPUT_COUNT",
  "spatial_coherence_score": 0.0,
  "execution_summary": {
    "selected_task": "CHANGE_ANALYSIS",
    "tool_used": "bi_temporal_change",
    "input_scope": "3 image(s)",
    "num_images_processed": 3,
    "agent_reasoning": "Validation failed at Stage 2/3: Tool 'CHANGE_ANALYSIS' requires between 2 and 2 images, but 3 image(s) were supplied."
  },
  "execution_trace": {
    "trace_id": "38b4f622-641c-47db-8f72-c57a12cdf496",
    "timestamp": "2026-10-14T19:56:02.489928",
    "query": "Analyze these three images.",
    "selected_task": "CHANGE_ANALYSIS",
    "task_confidence": 0.04,
    "input_checks": {
      "num_images": 3,
      "detected_modalities": [
        "OPTICAL",
        "OPTICAL",
        "OPTICAL"
      ],
      "detected_formats": [
        "PNG",
        "PNG",
        "PNG"
      ],
      "has_georeferencing": false,
      "benchmark_mode": true
    },
    "tools_called": [
      {
        "tool_id": "bi_temporal_change",
        "task_name": "CHANGE_ANALYSIS",
        "version": "1.0.0",
        "status": "REJECTED",
        "latency_ms": 99.64,
        "parameters_used": {},
        "summary": "Invalid image count for CHANGE_ANALYSIS: received 3, expected 2-2."
      }
    ],
    "per_step_latency_ms": {
      "input_validation": 99.64
    },
    "total_latency_ms": 99.64,
    "outputs_summary": {
      "status": "rejected",
      "error_code": "INVALID_INPUT_COUNT",
      "visual_evidence_count": 0,
      "geojson_features": 0
    },
    "confidence": 0.04,
    "confidence_derivation": "Calibrated rejection uncertainty: 0.04 derived from guardrail constraint INVALID_INPUT_COUNT.",
    "valid_sequence": false,
    "is_invalid_call": true,
    "structured_error": {
      "status": "error",
      "error_code": "INVALID_INPUT_COUNT",
      "error_type": "InputCountError",
      "message": "Invalid image count for CHANGE_ANALYSIS: received 3, expected 2-2.",
      "reason": "Tool 'CHANGE_ANALYSIS' requires between 2 and 2 images, but 3 image(s) were supplied.",
      "suggested_fix": "Provide exactly 2 image(s) for CHANGE_ANALYSIS.",
      "invalid_parameter": null,
      "details": {
        "received_count": 3,
        "expected_min": 2,
        "expected_max": 2
      }
    }
  },
  "structured_error": {
    "status": "error",
    "error_code": "INVALID_INPUT_COUNT",
    "error_type": "InputCountError",
    "message": "Invalid image count for CHANGE_ANALYSIS: received 3, expected 2-2.",
    "reason": "Tool 'CHANGE_ANALYSIS' requires between 2 and 2 images, but 3 image(s) were supplied.",
    "suggested_fix": "Provide exactly 2 image(s) for CHANGE_ANALYSIS.",
    "invalid_parameter": null,
    "details": {
      "received_count": 3,
      "expected_min": 2,
      "expected_max": 2
    }
  },
  "geo_metadata": null,
  "geojson_data": null
}
```

## 8. HARDCODED OR HEURISTIC CODE STILL ACTIVE

### 8.1 Grep for `"confidence" : 0.[0-9]+` and `confidence = 0.[0-9]+` in backend/
```
=== GREP 1: "confidence" : 0.xxx ===
[NONE FOUND]

=== GREP 2: confidence = 0.xxx ===
backend/query_understanding.py:367: confidence=0.10,
```
*(Note: `confidence=0.10` exists at `backend/query_understanding.py:367` inside an exception handler fallback block.)*

### 8.2 Hardcoded Place Names and Suspicious Vocabulary in backend/
```
=== SEARCHING BACKEND FOR TARGET WORDS: 'Tactical', 'Zero FPR', 'millimeter', '100%', Place Names ===
backend/answer_generator.py:383 [Match: 'Tactical']: "\n\n---\n\n### 3. Tactical GIS Interoperability\n"
backend/answer_generator.py:602 [Match: 'Tactical']: # 4. Tactical Evidence Note
backend/answer_generator.py:604 [Match: 'Tactical']: "### 4. 🗺️ Tactical Evidence & GIS Readiness\n"
backend/model_interfaces.py:620 [Match: 'Mumbai']: # D) Coastal silty / turbid water (e.g. Mumbai harbor / bays with sediment):
backend/model_interfaces.py:889 [Match: 'tactical']: # Find major water sectors and draw tactical bounding vectors
backend/model_interfaces.py:966 [Match: 'tactical']: # Find major land sectors and add tactical bounding boxes + labels
backend/model_interfaces.py:1233 [Match: 'Tactical']: f"• **Tactical GIS Observation:** Port navigational waters and berths highlighted in cyan in the visual evidence overlay."
backend/model_interfaces.py:1272 [Match: 'Mumbai']: loc_name = "South Mumbai Metropolitan Peninsula (Back Bay, Arabian Sea, Port of Mumbai), Maharashtra, India"
backend/model_interfaces.py:1274 [Match: 'Uttarakhand']: loc_name = "Garhwal / Himalayan Valley Basin, Uttarakhand, India"
backend/model_interfaces.py:1276 [Match: 'Delhi']: loc_name = "National Capital Region (Delhi/NCR), India"
backend/model_interfaces.py:1280 [Match: 'Mumbai']: loc_name = "South Mumbai / Western Indian Coastal Harbor & Peninsula Region"
backend/model_interfaces.py:1332 [Match: 'tactical']: f"• In the visual evidence panel, **water bodies are illuminated in luminous cyan-blue** with gold port structures..."
backend/model_interfaces.py:1416 [Match: 'Tactical']: f"🌐 **Tactical Remote Sensing Intelligence Assessment**\n\n"
backend/model_interfaces.py:1720 [Match: 'Tactical']: "mission": "SatQuery AI Tactical Grounding & Quantification",
backend/model_interfaces.py:1765 [Match: 'Tactical']: f"📍 **Tactical Spatial Delineation & Surface Area Quantification**\n\n"
backend/model_interfaces.py:1775 [Match: 'tactical']: f"• **Visual Evidence Verification:** In the evidence canvas on the right, the delineated target is highlighted..."
backend/model_interfaces.py:1777 [Match: 'Tactical']: f"### 3. Tactical GIS Interoperability\n"
backend/model_interfaces.py:1957 [Match: 'tactical']: # Draw high-visibility tactical bounding box
backend/model_interfaces.py:2065 [Match: 'tactical']: f"• All detected change zones are marked with **tactical red boundary boxes** on the post-event observation."
backend/model_interfaces.py:2097 [Match: 'tactical']: f"• In the visual evidence canvas on the right, all change zones are demarcated with **red tactical boxes**..."
backend/model_interfaces.py:2327 [Match: 'tactical']: # Annotate fused canvas with tactical boundary contours
backend/model_interfaces.py:2444 [Match: 'Tactical']: f"### 3. Tactical Visual Evidence & Co-Registration Verification\n"
backend/sample_missions.py:3 [Match: 'tactical']: Generates realistic tactical remote sensing image pairs and metadata
backend/sample_missions.py:15 [Match: 'Uttarakhand']: # 1. Uttarakhand Flash Flood Scenario (Bi-temporal Pair)
backend/sample_missions.py:47 [Match: 'Mumbai']: # 2. Mumbai Port Recon (Cartosat-2S High-Res Optical)
backend/sample_missions.py:122 [Match: 'uttarakhand']: "id": "uttarakhand_flood",
backend/sample_missions.py:125 [Match: 'Uttarakhand']: "location": "Rishi Ganga Valley, Uttarakhand (30.41° N, 79.73° E)",
backend/sample_missions.py:143 [Match: 'mumbai']: "id": "mumbai_port_recon",
backend/sample_missions.py:146 [Match: 'Mumbai']: "location": "Jawaharlal Nehru Port, Navi Mumbai (18.95° N, 72.95° E)",
backend/sample_missions.py:165 [Match: '100%']: "description": "Cross-modal pair demonstrating 100% cloud penetration via RISAT-1 C-band microwave radar."
backend/test_change_pipeline.py:22 [Match: 'uttarakhand']: def uttarakhand_pair():
backend/test_cross_modal.py:22 [Match: 'mumbai']: p_opt = DEMO_DIR / "mumbai_cartosat2s_optical.png"
backend/test_query_routing.py:376 [Match: '100%']: """Specific tests for the 5 target user queries to ensure 100% routing & planning precision."""
backend/tiling.py:78 [Match: '100%']: Guarantees 100% coverage up to the right and bottom image borders.
```

**Key Heuristic Findings in `backend/model_interfaces.py`:**
- **Lines 1279–1280:** `elif w_pct > 40.0 and built_pct > 20.0: loc_name = "South Mumbai / Western Indian Coastal Harbor & Peninsula Region"`
  Hardcodes the location string to Mumbai based purely on heuristic land/water pixel ratios, without georeferencing.
- **Lines 1271–1276:** Geographic coordinates are mapped to fixed place names via latitude/longitude if-conditions (`South Mumbai`, `Garhwal / Himalayan Valley Basin, Uttarakhand`, `National Capital Region (Delhi/NCR)`).

### 8.3 Tools Marked `fallback_heuristic` and Trigger Conditions


| Tool ID | Task Name | Marked `fallback_heuristic`? | Condition Triggering Fallback |
| :--- | :--- | :---: | :--- |
| `single_image_vqa` | `SINGLE_IMAGE_VQA` | **NO** (Primary: `SatQuery-RS-VLM-v1.0`) | Triggers fallback to `OpenCV Spectral Discriminator` if checkpoint is missing or `ENABLE_ML_MODELS!=true`. |
| `land_cover_segmentation` | `LAND_COVER_SEGMENTATION` | **NO** (Primary: `SatSegNet-v1.0 ONNX INT8`) | Triggers fallback to `OpenCV Spectral Discriminator` if ONNX runtime fails and PyTorch `.pth` checkpoint is missing. |
| `bi_temporal_change` | `CHANGE_ANALYSIS` | **NO** (Primary: `SatQueryChangeNet`) | Triggers fallback to `SSIM Structural Difference Engine` if `change_net.pth` is missing or fails to load. |
| `optical_sar_fusion` | `CROSS_MODAL_EXTRACTION` | **NO** (Primary: `SatQueryOpticalSARNet`) | Triggers fallback to `IHS Cross-Modal Fusion Engine` if `optical_sar_net.pth` is missing or fails to load. |
| `single_image_grounding` | `SINGLE_IMAGE_GROUNDING` | **NO** (Primary: `SatQuery Grounding Head`) | Triggers fallback to `spectral thresholding & contour delineation` when bounding vector localization is requested without trained grounding head. |
| `heuristic_fallback` | `FALLBACK_HEURISTIC` | **YES** (`is_heuristic_fallback=True`) | Registered explicitly as safety fallback for unhandled or general exploratory queries. |

## 9. INPUT HANDLING

### 9.1 Accepted vs Rejected Formats

- **Accepted Formats:** GeoTIFF (`.tif`, `.geotiff`), TIFF (`.tiff`), JP2 (`.jp2`), PNG (`.png`), JPEG (`.jpg`, `.jpeg`), WebP (`.webp`), AVIF (`.avif`), BMP (`.bmp`).
- **Benchmark Mode Policy (`benchmark_mode=True`):** Accepts standard PNG, JPEG, and TIFF benchmark images without georeferencing.
- **Production Mode Policy (`benchmark_mode=False`):** Strictly rejects non-georeferenced imagery (PNG, JPEG, unreferenced TIFF) with error code `UNSUPPORTED_FORMAT`.

#### Exact Error JSON for PNG upload with benchmark_mode=false
```json
{
  "status": "rejected",
  "answer": "⚠️ **Request Validation Error (UNSUPPORTED_FORMAT)**\n\nStandard raster format 'PNG' is not permitted in production georeferenced mode.\n\n• **Reason:** Operational ISRO queries require calibrated 16-bit GeoTIFF imagery with CRS/transform metadata.\n• **Action Required:** Upload a georeferenced GeoTIFF (.tif/.geotiff) or enable benchmark mode toggle for public PNG/JPEG datasets.",
  "visual_evidence": [],
  "confidence": 0.04,
  "compatibility_status": "REJECTED_UNSUPPORTED_FORMAT",
  "spatial_coherence_score": 0.0,
  "execution_summary": {
    "selected_task": "LAND_COVER_SEGMENTATION",
    "tool_used": "land_cover_segmentation",
    "input_scope": "1 image(s)",
    "num_images_processed": 1,
    "agent_reasoning": "Validation failed at Stage 2/3: Operational ISRO queries require calibrated 16-bit GeoTIFF imagery with CRS/transform metadata."
  },
  "execution_trace": {
    "trace_id": "5fc24308-6b28-4524-bf3d-050652b91100",
    "timestamp": "2026-10-14T20:12:12.847259",
    "query": "Land cover",
    "selected_task": "LAND_COVER_SEGMENTATION",
    "task_confidence": 0.04,
    "input_checks": {
      "num_images": 1,
      "detected_modalities": [
        "OPTICAL"
      ],
      "detected_formats": [
        "PNG"
      ],
      "has_georeferencing": false,
      "benchmark_mode": false
    },
    "tools_called": [
      {
        "tool_id": "land_cover_segmentation",
        "task_name": "LAND_COVER_SEGMENTATION",
        "version": "1.0.0",
        "status": "REJECTED",
        "latency_ms": 47.42,
        "parameters_used": {},
        "summary": "Standard raster format 'PNG' is not permitted in production georeferenced mode."
      }
    ],
    "per_step_latency_ms": {
      "input_validation": 47.42
    },
    "total_latency_ms": 47.42,
    "outputs_summary": {
      "status": "rejected",
      "error_code": "UNSUPPORTED_FORMAT",
      "visual_evidence_count": 0,
      "geojson_features": 0
    },
    "confidence": 0.04,
    "confidence_derivation": "Calibrated rejection uncertainty: 0.04 derived from guardrail constraint UNSUPPORTED_FORMAT.",
    "valid_sequence": false,
    "is_invalid_call": true,
    "structured_error": {
      "status": "error",
      "error_code": "UNSUPPORTED_FORMAT",
      "error_type": "UnsupportedFormatError",
      "message": "Standard raster format 'PNG' is not permitted in production georeferenced mode.",
      "reason": "Operational ISRO queries require calibrated 16-bit GeoTIFF imagery with CRS/transform metadata.",
      "suggested_fix": "Upload a georeferenced GeoTIFF (.tif/.geotiff) or enable benchmark mode toggle for public PNG/JPEG datasets.",
      "invalid_parameter": null,
      "details": {
        "detected_format": "PNG",
        "benchmark_mode": false
      }
    }
  },
  "structured_error": {
    "status": "error",
    "error_code": "UNSUPPORTED_FORMAT",
    "error_type": "UnsupportedFormatError",
    "message": "Standard raster format 'PNG' is not permitted in production georeferenced mode.",
    "reason": "Operational ISRO queries require calibrated 16-bit GeoTIFF imagery with CRS/transform metadata.",
    "suggested_fix": "Upload a georeferenced GeoTIFF (.tif/.geotiff) or enable benchmark mode toggle for public PNG/JPEG datasets.",
    "invalid_parameter": null,
    "details": {
      "detected_format": "PNG",
      "benchmark_mode": false
    }
  },
  "geo_metadata": null,
  "geojson_data": null
}
```

#### Exact Error JSON for Corrupt File Upload
```json
{
  "status": "error",
  "message": "Unsupported or corrupted image format. Please ensure the file is a valid satellite or visual image (GeoTIFF, TIFF, JP2, PNG, JPEG, WebP, AVIF, BMP)."
}
```

#### Exact Error JSON for Oversized File (>100MB)
```json
{
  "status": "error",
  "message": "Unsupported or corrupted image format. Please ensure the file is a valid satellite or visual image (GeoTIFF, TIFF, JP2, PNG, JPEG, WebP, AVIF, BMP)."
}
```

### 9.2 Co-Registration Algorithms & Thresholds
- **File & Function:** `backend/model_interfaces.py:check_spatial_compatibility` (lines 389–535)
- **Same-Sensor Optical Pairs:**
  1. Keypoint Extraction: SIFT (`nfeatures=3000, contrastThreshold=0.03, edgeThreshold=10`), falling back to ORB (`nfeatures=2000`).
  2. Matching: BFMatcher with Lowe's ratio test (`distance < 0.75 * second_distance`).
  3. Transformation: RANSAC Homography (`cv2.findHomography(src, dst, cv2.RANSAC, 5.0)`).
  4. Disparity Rejection Threshold: `< 15 inliers` OR `inlier_ratio < 0.30` (emits `Spatial Disparity` diagnostic).
- **Cross-Modal Optical-SAR Pairs:**
  1. Modality Alignment: Bypasses keypoints (which fail across microwave and optical spectra).
  2. Metrics: 32-bin 2D Normalized Mutual Information (NMI) + 3x3 Sobel Gradient Correlation (`grad_corr`).
  3. Disparity Rejection Threshold: `nmi < 0.010` OR `grad_corr <= 0.0` (emits `Cross-Modal Disparity Detected`).
- **Georeferenced Pairs (GeoTIFF):**
  - Requires CRS equality and Bounding Box intersection IoU >= 0.15 (15%). If IoU < 0.15, rejected with `GeoTIFF Bounding Box IoU < 15%`.

### 9.3 Modality Detection Algorithm
- **File & Function:** `backend/model_interfaces.py:detect_image_modality` (lines 261–330)
- **Algorithm:**
  1. Rasterio inspection: checks band count, data type, and tags (`desc_str` and `tags_str` for `"sar"`, `"vv"`, `"vh"`, `"sigma0"`, `"risat"`).
  2. Multi-spectral: `band_count > 3` -> `MULTISPECTRAL` (Cartosat / Sentinel-2).
  3. Scientific SAR: single/dual-band float32/int16/uint16 with radar tags -> `SAR`.
  4. Filename heuristics: fallback substring matching for `"sar"`, `"radar"`, `"risat"`, `"sentinel1"`.
  5. Default fallback: 3-band uint8 -> `OPTICAL`.

### 9.4 Real Measured Latency Table (from `reports/latency.json`)
- **Hardware:** Intel 12-Core Host CPU, Torch 2.13.0+cpu, No CUDA GPU.
- **Iterations:** 15 iterations per tool.
- **Batch / Tile Size:** Single tile (256x256 or 512x512).

| Specialist Tool | P50 Latency (ms) | Mean Latency (ms) | P95 Latency (ms) | Throughput (QPS) |
| :--- | :---: | :---: | :---: | :---: |
| Optical-SAR Fusion | 7.07 | 7.16 | 8.59 | 139.75 |
| Single-Image Grounding | 13.24 | 13.26 | 14.13 | 75.44 |
| Raster Ingestion & Decode | 13.29 | 13.31 | 14.58 | 75.13 |
| Single-Image Captioning | 13.72 | 14.13 | 16.35 | 70.79 |
| Land Cover Segmentation | 13.76 | 14.97 | 21.31 | 66.80 |
| Single-Image VQA | 15.01 | 17.71 | 29.17 | 56.46 |
| Spatial Compatibility Verifier | 27.76 | 28.54 | 34.66 | 35.04 |
| Bi-Temporal Change Detection | 51.76 | 50.55 | 58.05 | 19.78 |
| Large Raster Tiling & Stitching (2048px) | 229.29 | 236.86 | 274.93 | 4.22 |

## 10. FRONTEND

### 10.1 Frontend Build Output (`npm run build`)
```
> frontend@0.1.0 build
> next build

▲ Next.js 16.3.4 (Turbopack)
- Environments: .env.local
✓ Running next.config.ts took 70ms

  Creating an optimized production build ...
✓ Compiled successfully in 3.3s
  Skipping validation of types
  Finished TypeScript config validation in 16ms ...
  Collecting page data using 11 workers ...
  Generating static pages using 11 workers (0/11) ...
  Generating static pages using 11 workers (2/11) 
  Generating static pages using 11 workers (5/11) 
  Generating static pages using 11 workers (8/11) 
✓ Generating static pages using 11 workers (11/11) in 2.4s
  Finalizing page optimization ...

Route (app)
┌ ○ /
├ ○ /_not-found
├ ○ /acquire
├ ○ /dashboard
├ ○ /faq
├ ○ /gallery
├ ○ /manifest.webmanifest
├ ○ /offline
└ ○ /query


○  (Static)  prerendered as static content
```

### 10.2 Frontend Lint Output (`npm run lint`)
```
> frontend@0.1.0 lint
> eslint

D:\sih26167\frontend\public\sw.js
  126:18  warning  'err' is defined but never used  @typescript-eslint/no-unused-vars

D:\sih26167\frontend\src\app\dashboard\page.tsx
   97:57   error    Unexpected any. Specify a different type                         @typescript-eslint/no-explicit-any
  100:10   warning  'loading' is assigned a value but never used                     @typescript-eslint/no-unused-vars
  295:53   error    Unexpected any. Specify a different type                         @typescript-eslint/no-explicit-any
  425:168  error    Unexpected any. Specify a different type                         @typescript-eslint/no-explicit-any
  515:168  error    Unexpected any. Specify a different type                         @typescript-eslint/no-explicit-any
  578:168  error    Unexpected any. Specify a different type                         @typescript-eslint/no-explicit-any
  635:168  error    Unexpected any. Specify a different type                         @typescript-eslint/no-explicit-any
  753:28   error    `"` can be escaped with `&quot;`, `&ldquo;`, `&#34;`, `&rdquo;`  react/no-unescaped-entities
  753:53   error    `"` can be escaped with `&quot;`, `&ldquo;`, `&#34;`, `&rdquo;`  react/no-unescaped-entities

D:\sih26167\frontend\src\app\offline\page.tsx
  12:5  error  Error: Calling setState synchronously within an effect can trigger cascading renders  react-hooks/set-state-in-effect

D:\sih26167\frontend\src\app\page.tsx
  6:8  warning  'BackgroundSlideshow' is defined but never used  @typescript-eslint/no-unused-vars

D:\sih26167\frontend\src\app\query\page.tsx
    10:12  error    Unexpected any. Specify a different type  @typescript-eslint/no-explicit-any
    15:8   warning  'BackgroundSlideshow' is defined but never used  @typescript-eslint/no-unused-vars
    16:8   warning  'FadeInScroll' is defined but never used  @typescript-eslint/no-unused-vars
    43:51  error    Unexpected any. Specify a different type  @typescript-eslint/no-explicit-any
   198:56  error    Unexpected any. Specify a different type  @typescript-eslint/no-explicit-any
   337:54  error    Unexpected any. Specify a different type  @typescript-eslint/no-explicit-any
   345:7   error    Error: Calling setState synchronously within an effect can trigger cascading renders  react-hooks/set-state-in-effect
   399:68  error    Error: Cannot call impure function during render  react-hooks/purity
   553:7   error    Error: Calling setState synchronously within an effect can trigger cascading renders  react-hooks/set-state-in-effect
   556:6   warning  React Hook useEffect has a missing dependency: 'loadingStages.length'. Either include it or remove the dependency array  react-hooks/exhaustive-deps
   645:19  error    Unexpected any. Specify a different type  @typescript-eslint/no-explicit-any
   686:19  error    Unexpected any. Specify a different type  @typescript-eslint/no-explicit-any
   697:64  error    Unexpected any. Specify a different type  @typescript-eslint/no-explicit-any
   842:36  error    Unexpected any. Specify a different type  @typescript-eslint/no-explicit-any
   949:11  error    Do not use an `<a>` element to navigate to `/`. Use `<Link />` from `next/link` instead.  @next/next/no-html-link-for-pages
   998:11  error    Do not use an `<a>` element to navigate to `/`. Use `<Link />` from `next/link` instead.  @next/next/no-html-link-for-pages
  1123:27  warning  Using `<img>` could result in slower LCP and higher bandwidth. Consider using `<Image />`  @next/next/no-img-element
  1183:53  error    Unexpected any. Specify a different type  @typescript-eslint/no-explicit-any
  1215:52  error    Unexpected any. Specify a different type  @typescript-eslint/no-explicit-any
  1330:35  error    `"` can be escaped with `&quot;`, `&ldquo;`, `&#34;`, `&rdquo;`  react/no-unescaped-entities
  1330:39  error    `"` can be escaped with `&quot;`, `&ldquo;`, `&#34;`, `&rdquo;`  react/no-unescaped-entities
  1474:98  error    Unexpected any. Specify a different type  @typescript-eslint/no-explicit-any
  1506:72  error    Unexpected any. Specify a different type  @typescript-eslint/no-explicit-any
  1568:58  error    Unexpected any. Specify a different type  @typescript-eslint/no-explicit-any
  1575:27  warning  Using `<img>` could result in slower LCP and higher bandwidth. Consider using `<Image />`  @next/next/no-img-element
  1653:39  warning  'bbox' is defined but never used  @typescript-eslint/no-unused-vars
  1681:13  warning  Using `<img>` could result in slower LCP and higher bandwidth. Consider using `<Image />`  @next/next/no-img-element

D:\sih26167\frontend\src\components\AuthModal.tsx
   17:5   warning  'mapCount' is assigned a value but never used  @typescript-eslint/no-unused-vars
   19:5   warning  'maxFreeMapQueries' is assigned a value but never used  @typescript-eslint/no-unused-vars
   31:7   error    Error: Calling setState synchronously within an effect can trigger cascading renders  react-hooks/set-state-in-effect
   41:7   error    Error: Calling setState synchronously within an effect can trigger cascading renders  react-hooks/set-state-in-effect
   50:55  error    Unexpected any. Specify a different type  @typescript-eslint/no-explicit-any
   52:37  error    Unexpected any. Specify a different type  @typescript-eslint/no-explicit-any
   57:34  error    Unexpected any. Specify a different type  @typescript-eslint/no-explicit-any
  107:6   warning  React Hook useEffect has a missing dependency: 'loginWithGoogle'. Either include it or remove the dependency array  react-hooks/exhaustive-deps
  123:53  error    Unexpected any. Specify a different type  @typescript-eslint/no-explicit-any
  125:35  error    Unexpected any. Specify a different type  @typescript-eslint/no-explicit-any
  126:50  error    Unexpected any. Specify a different type  @typescript-eslint/no-explicit-any

D:\sih26167\frontend\src\components\GeoJSONMapOverlay.tsx
   5:16  error    Unexpected any. Specify a different type  @typescript-eslint/no-explicit-any
  30:33  error    Unexpected any. Specify a different type  @typescript-eslint/no-explicit-any
  40:34  error    Unexpected any. Specify a different type  @typescript-eslint/no-explicit-any
  50:19  error    Unexpected any. Specify a different type  @typescript-eslint/no-explicit-any
  99:16  warning  'e' is defined but never used  @typescript-eslint/no-unused-vars

D:\sih26167\frontend\src\components\GlobeCarousel3d.tsx
    8:10  warning  'hash01' is defined but never used  @typescript-eslint/no-unused-vars
   12:10  warning  'smoothstep' is defined but never used  @typescript-eslint/no-unused-vars
   16:10  warning  'clamp' is defined but never used  @typescript-eslint/no-unused-vars
   19:10  warning  'wrap' is defined but never used  @typescript-eslint/no-unused-vars
   53:71  error    Unexpected any. Specify a different type  @typescript-eslint/no-explicit-any
   60:7   error    'la' is never reassigned. Use 'const' instead  prefer-const
   72:33  error    Unexpected any. Specify a different type  @typescript-eslint/no-explicit-any
   90:52  error    Unexpected any. Specify a different type  @typescript-eslint/no-explicit-any
  101:28  error    Unexpected any. Specify a different type  @typescript-eslint/no-explicit-any
  138:12  error    Unexpected any. Specify a different type  @typescript-eslint/no-explicit-any
  140:13  error    Unexpected any. Specify a different type  @typescript-eslint/no-explicit-any
  140:26  error    Unexpected any. Specify a different type  @typescript-eslint/no-explicit-any
  147:13  error    Unexpected any. Specify a different type  @typescript-eslint/no-explicit-any
  169:24  error    Unexpected any. Specify a different type  @typescript-eslint/no-explicit-any
  170:25  error    Unexpected any. Specify a different type  @typescript-eslint/no-explicit-any

D:\sih26167\frontend\src\components\Model3dViewer.tsx
  10:8  warning  'useFrame' is defined but never used  @typescript-eslint/no-unused-vars

D:\sih26167\frontend\src\components\NetworkStatusBanner.tsx
  13:5  error  Error: Calling setState synchronously within an effect can trigger cascading renders  react-hooks/set-state-in-effect

D:\sih26167\frontend\src\components\SphereOrbit.tsx
  145:113  error  Passing a ref to a function may read its value during render  react-hooks/refs

D:\sih26167\frontend\src\components\SwipeSlider.tsx
  83:9   warning  Using `<img>` could result in slower LCP and higher bandwidth. Consider using `<Image />`  @next/next/no-img-element
  94:11  warning  Using `<img>` could result in slower LCP and higher bandwidth. Consider using `<Image />`  @next/next/no-img-element

D:\sih26167\frontend\src\components\TiffPreview.tsx
  45:36  error  Unexpected any. Specify a different type  @typescript-eslint/no-explicit-any
  57:36  error  Unexpected any. Specify a different type  @typescript-eslint/no-explicit-any
  58:36  error  Unexpected any. Specify a different type  @typescript-eslint/no-explicit-any
  59:36  error  Unexpected any. Specify a different type  @typescript-eslint/no-explicit-any

D:\sih26167\frontend\src\context\AuthContext.tsx
   55:9   error    Error: Calling setState synchronously within an effect can trigger cascading renders  react-hooks/set-state-in-effect
   94:16  warning  'e' is defined but never used  @typescript-eslint/no-unused-vars
  105:16  warning  'e' is defined but never used  @typescript-eslint/no-unused-vars
  120:14  warning  'e' is defined but never used  @typescript-eslint/no-unused-vars
  134:14  warning  'e' is defined but never used  @typescript-eslint/no-unused-vars
  149:14  warning  'e' is defined but never used  @typescript-eslint/no-unused-vars
  164:14  warning  'e' is defined but never used  @typescript-eslint/no-unused-vars
  173:14  warning  'e' is defined but never used  @typescript-eslint/no-unused-vars

✖ 158 problems (116 errors, 42 warnings)
  5 errors and 0 warnings potentially fixable with the `--fix` option.
```

### 10.3 Pages & Routes Summary

| Route | Page File | Function & Display Content | Metrics Displayed | Metric Source |
| :--- | :--- | :--- | :--- | :--- |
| `/` | `frontend/src/app/page.tsx` | Landing page, 3D rotating globe, mission CTA, capability overview. | Feature bullet tags | Static copy |
| `/query` | `frontend/src/app/query/page.tsx` | Mission control workspace: upload, preflight checks, natural language dialogue, visual evidence, GeoJSON map, PDF export. | Dynamic response confidence, latency, surface area (ha/km²) | Live `/api/query` response payload |
| `/dashboard` | `frontend/src/app/dashboard/page.tsx` | Audited telemetry dashboard: 6 tabs (Overview radar, Routing 167 cases, VLM benchmarks, Optical-SAR, Latency, Live reports JSON explorer). | 91.02% routing acc, 99.31% valid seq, 76.67% VRSBench, 100% CDVQA, 7.07ms latency | Live `/api/reports` backend API |
| `/acquire` | `frontend/src/app/acquire/page.tsx` | Interactive satellite map, bounding box / polygon AOI selector. | Coordinates, area in km² | Leaflet canvas computation |
| `/gallery` | `frontend/src/app/gallery/page.tsx` | 3D space museum and satellite imagery gallery. | Sensor metadata | Static imagery cards |
| `/faq` | `frontend/src/app/faq/page.tsx` | Technical architecture questions, sensor specifications. | Technical specs | FAQ items array |
| `/offline` | `frontend/src/app/offline/page.tsx` | Offline service worker fallback page for PWA. | None | Static PWA fallback |

### 10.4 PDF Audit Export Contents
- **Engine:** jsPDF client-side vector/raster generation (`frontend/src/app/query/page.tsx:720-915`).
- **Sections generated:**
  1. Header Banner: `OPERATIONAL MISSION AUDIT | ISRO/SAC & BENCHMARK SUITE` (Mission timestamp, Operator, Problem Statement PS 26167)
  2. Section 1: Executive Mission Findings & Natural Language Dialogue
  3. Section 2: Input Data & Sensor Metrology (File Name, Modality, Bit Depth, Dimensions, CRS & Bounds)
  4. Section 3: Pairwise Spatial Compatibility & Co-Registration Audit (Verification Status, Coherence Score)
  5. Section 4: Quantitative Surface Measurements & Land-Cover Breakdown (Area in Hectares & Sq Km)
  6. Section 5: Vector Delineation & RFC 7946 GeoJSON Feature Geometry (Polygon coordinates, bounding vertices)
  7. Section 6: Verified Benchmark & Claims Audit Provenance (Telemetry verification statement, routing benchmark 91.02%)
  8. Page Footer: `CONFIDENTIAL & PROPRIETARY — SATQUERY AI CORE ENGINE | ISRO/SAC EVALUATION PROTOCOL`

## 11. TESTS AND CI

### 11.1 Full Pytest Run Output
```
============================= test session starts =============================
platform win32 -- Python 3.10.11, pytest-9.1.1, pluggy-1.6.0 -- D:\sih26167\backend\venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: D:\sih26167
plugins: anyio-4.14.2, asyncio-1.4.0
asyncio: mode=strict, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 81 items

backend/test_change_pipeline.py::test_identical_images_rejected PASSED   [  1%]
backend/test_change_pipeline.py::test_disparate_images_rejected PASSED   [  2%]
backend/test_change_pipeline.py::test_uttarakhand_demo_pair_passes PASSED [  3%]
backend/test_change_pipeline.py::test_decode_palette_image_preserves_colors PASSED [  4%]
backend/test_cross_modal.py::test_modality_detection_accuracy PASSED     [  6%]
backend/test_cross_modal.py::test_optical_sar_true_pair_passes PASSED    [  7%]
backend/test_cross_modal.py::test_optical_sar_disparate_pair_rejected PASSED [  8%]
backend/test_cross_modal.py::test_cross_modal_analysis_execution PASSED  [  9%]
backend/test_cross_modal.py::test_single_image_sar_contradiction_guardrail PASSED [ 11%]
backend/test_input_robustness.py::test_16bit_geotiff_ingestion PASSED    [ 12%]
backend/test_input_robustness.py::test_float32_sar_geotiff_ingestion PASSED [ 13%]
backend/test_input_robustness.py::test_multi_band_4band_geotiff PASSED   [ 14%]
backend/test_input_robustness.py::test_nodata_handling PASSED            [ 16%]
backend/test_input_robustness.py::test_empty_blank_geotiff_rejection PASSED [ 17%]
backend/test_input_robustness.py::test_corrupt_bytes_rejection PASSED    [ 18%]
backend/test_input_robustness.py::test_sliding_window_tiling_coverage PASSED [ 19%]
backend/test_input_robustness.py::test_sliding_window_stitching_accuracy PASSED [ 20%]
backend/test_input_robustness.py::test_sliding_window_detection_remapping_and_nms PASSED [ 22%]
backend/test_input_robustness.py::test_benchmark_mode_format_enforcement PASSED [ 23%]
backend/test_query_routing.py::TestRegressionBuildingLocalization::test_locate_buildings_not_land_cover PASSED [ 24%]
backend/test_query_routing.py::TestRegressionBuildingLocalization::test_locate_buildings_targets_extracted PASSED [ 25%]
backend/test_query_routing.py::TestRegressionBuildingLocalization::test_locate_buildings_plan_uses_detection PASSED [ 27%]
backend/test_query_routing.py::TestObjectLocalization::test_locate_buildings PASSED [ 28%]
backend/test_query_routing.py::TestObjectLocalization::test_find_storage_facilities PASSED [ 29%]
backend/test_query_routing.py::TestObjectLocalization::test_find_bridges PASSED [ 30%]
backend/test_query_routing.py::TestObjectLocalization::test_detect_airports PASSED [ 32%]
backend/test_query_routing.py::TestObjectLocalization::test_military_infrastructure PASSED [ 33%]
backend/test_query_routing.py::TestObjectLocalization::test_find_warehouses PASSED [ 34%]
backend/test_query_routing.py::TestObjectLocalization::test_locate_vehicles PASSED [ 35%]
backend/test_query_routing.py::TestObjectLocalization::test_find_construction_sites PASSED [ 37%]
backend/test_query_routing.py::TestObjectLocalization::test_identify_residential PASSED [ 38%]
backend/test_query_routing.py::TestObjectCounting::test_count_buildings PASSED [ 39%]
backend/test_query_routing.py::TestObjectCounting::test_count_buildings_variant PASSED [ 40%]
backend/test_query_routing.py::TestObjectCounting::test_number_of_ships PASSED [ 41%]
backend/test_query_routing.py::TestLandCover::test_land_cover_classification PASSED [ 43%]
backend/test_query_routing.py::TestLandCover::test_agricultural_percentage PASSED [ 44%]
backend/test_query_routing.py::TestLandCover::test_built_up_area PASSED  [ 45%]
backend/test_query_routing.py::TestLandCover::test_distribution_query PASSED [ 46%]
backend/test_query_routing.py::TestLandCover::test_built_area_plan_uses_land_cover PASSED [ 48%]
backend/test_query_routing.py::TestWaterBody::test_find_water_bodies PASSED [ 49%]
backend/test_query_routing.py::TestWaterBody::test_identify_rivers PASSED [ 50%]
backend/test_query_routing.py::TestWaterBody::test_locate_rivers PASSED  [ 51%]
backend/test_query_routing.py::TestVegetation::test_dense_vegetation PASSED [ 53%]
backend/test_query_routing.py::TestVegetation::test_agricultural_fields PASSED [ 54%]
backend/test_query_routing.py::TestRoads::test_find_roads PASSED         [ 55%]
backend/test_query_routing.py::TestMultiObject::test_multi_object_query PASSED [ 56%]
backend/test_query_routing.py::TestMultiObject::test_multi_object_not_land_cover PASSED [ 58%]
backend/test_query_routing.py::TestSpatialRelationship::test_buildings_near_roads PASSED [ 59%]
backend/test_query_routing.py::TestDirection::test_northern_query PASSED [ 60%]
backend/test_query_routing.py::TestDirection::test_eastern_buildings PASSED [ 61%]
backend/test_query_routing.py::TestDirection::test_upper_right PASSED    [ 62%]
backend/test_query_routing.py::TestDirection::test_spatial_constraint_extracted PASSED [ 64%]
backend/test_query_routing.py::TestComparison::test_east_west_comparison PASSED [ 65%]
backend/test_query_routing.py::TestComparison::test_two_image_change PASSED [ 66%]
backend/test_query_routing.py::TestComparison::test_urban_expansion PASSED [ 67%]
backend/test_query_routing.py::TestSceneUnderstanding::test_what_is_visible PASSED [ 69%]
backend/test_query_routing.py::TestSceneUnderstanding::test_describe_image PASSED [ 70%]
backend/test_query_routing.py::TestSceneUnderstanding::test_general_overview PASSED [ 71%]
backend/test_query_routing.py::TestSceneUnderstanding::test_ambiguous PASSED [ 72%]
backend/test_query_routing.py::TestConfidence::test_clear_localization_high_confidence PASSED [ 74%]
backend/test_query_routing.py::TestConfidence::test_ambiguous_lower_confidence PASSED [ 75%]
backend/test_query_routing.py::TestConfidence::test_confidence_in_range PASSED [ 76%]
backend/test_query_routing.py::TestPlanCorrectness::test_count_plan_has_count_step PASSED [ 77%]
backend/test_query_routing.py::TestPlanCorrectness::test_spatial_relation_plan PASSED [ 79%]
backend/test_query_routing.py::TestPlanCorrectness::test_water_query_not_object_detection_primary PASSED [ 80%]
backend/test_query_routing.py::TestTargetFiveQuestions::test_query_1_describe_landcover_and_objects PASSED [ 81%]
backend/test_query_routing.py::TestTargetFiveQuestions::test_query_2_highlight_water_body PASSED [ 82%]
backend/test_query_routing.py::TestTargetFiveQuestions::test_query_3_what_changed_and_where PASSED [ 83%]
backend/test_query_routing.py::TestTargetFiveQuestions::test_query_4_optical_and_sar_fusion PASSED [ 85%]
backend/test_query_routing.py::TestTargetFiveQuestions::test_query_5_built_up_area_trend PASSED [ 86%]
backend/test_query_routing.py::TestOutOfDomain::test_stock_price_rejection PASSED [ 87%]
backend/test_query_routing.py::TestOutOfDomain::test_poem_rejection PASSED [ 88%]
backend/test_query_routing.py::TestOutOfDomain::test_code_rejection PASSED [ 90%]
backend/test_query_routing.py::TestOutOfDomain::test_general_chat_rejection PASSED [ 91%]
backend/test_tool_registry.py::test_registry_has_all_tools PASSED        [ 92%]
backend/test_tool_registry.py::test_format_detection PASSED              [ 93%]
backend/test_tool_registry.py::test_input_validation_count PASSED        [ 95%]
backend/test_tool_registry.py::test_input_validation_modality PASSED     [ 96%]
backend/test_tool_registry.py::test_parameter_validation PASSED          [ 97%]
backend/test_tool_registry.py::test_controller_trace_generation PASSED   [ 98%]
backend/test_tool_registry.py::test_controller_invalid_input_rejection PASSED [100%]

============================== warnings summary ===============================
backend/test_change_pipeline.py::test_disparate_images_rejected
backend/test_change_pipeline.py::test_uttarakhand_demo_pair_passes
backend/test_change_pipeline.py::test_decode_palette_image_preserves_colors
backend/test_cross_modal.py::test_modality_detection_accuracy
backend/test_cross_modal.py::test_optical_sar_true_pair_passes
backend/test_cross_modal.py::test_optical_sar_disparate_pair_rejected
backend/test_cross_modal.py::test_cross_modal_analysis_execution
backend/test_cross_modal.py::test_single_image_sar_contradiction_guardrail
backend/test_tool_registry.py::test_controller_invalid_input_rejection
  D:\sih26167\backend\venv\lib\site-packages\rasterio\io.py:140: NotGeoreferencedWarning: Dataset has no geotransform, gcps, or rpcs. The identity matrix will be returned.
    rd = DatasetReader(mempath, driver=driver, sharing=sharing, **kwargs)

backend/test_input_robustness.py::test_16bit_geotiff_ingestion
backend/test_input_robustness.py::test_float32_sar_geotiff_ingestion
backend/test_input_robustness.py::test_multi_band_4band_geotiff
backend/test_input_robustness.py::test_nodata_handling
backend/test_input_robustness.py::test_empty_blank_geotiff_rejection
  D:\sih26167\backend\venv\lib\site-packages\rasterio\transform.py:189: PendingDeprecationWarning: Use `@` matmul instead of `*` mul operator for matrix multiplication
    return Affine.translation(west, north) * Affine.scale(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
======================= 81 passed, 14 warnings in 8.76s =======================
Skipped / Xfail tests: 0 skipped, 0 xfail
```

### 11.2 Model Quality Tests (Beyond Routing & Schema)

| Test Name | File | What It Asserts |
| :--- | :--- | :--- |
| `test_disparate_images_rejected` | `backend/test_change_pipeline.py` | Asserts that non-overlapping pairs (Uttarakhand flood vs Bengaluru optical) yield `is_compatible=False`, `score <= 0.35`, and are rejected before execution. |
| `test_uttarakhand_demo_pair_passes` | `backend/test_change_pipeline.py` | Asserts that authentic bi-temporal pair (Uttarakhand Pre vs Post) yields `is_compatible=True`, `score >= 0.70`, and completes change mask generation. |
| `test_optical_sar_true_pair_passes` | `backend/test_cross_modal.py` | Asserts that Mumbai Optical + RISAT-1 SAR pair passes co-registration guardrail (`is_compatible=True`, `coherence_score >= 0.50`). |
| `test_optical_sar_disparate_pair_rejected` | `backend/test_cross_modal.py` | Asserts that disparate Bengaluru Optical + RISAT-1 SAR pair is rejected (`is_compatible=False`, `coherence_score <= 0.35`). |
| `test_single_image_sar_contradiction_guardrail` | `backend/test_cross_modal.py` | Asserts that attempting cross-modal analysis with a single optical image is rejected with `INVALID_INPUT_COUNT`. |
| `test_sliding_window_tiling_coverage` | `backend/test_input_robustness.py` | Asserts that `TilerStitcher` reconstructs 2048x2048 tiles with zero seam artifacts and complete border coverage. |

### 11.3 GitHub Actions CI (`.github/workflows/ci.yml`)
```yaml

name: SatQuery AI CI

on:
  push:
    branches: [ main, master ]
  pull_request:
    branches: [ main, master ]

jobs:
  backend-test:
    name: Backend Test Suite & Claims Audit
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: "pip"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          if [ -f backend/requirements.txt ]; then pip install -r backend/requirements.txt; fi
          pip install pytest pytest-asyncio httpx rasterio shapely opencv-python-headless

      - name: Run Claims & Provenance Auditor
        run: |
          python scripts/audit_claims.py

      - name: Run Backend Pytest Suite
        env:
          ENABLE_ML_MODELS: "false"
          TRANSFORMERS_OFFLINE: "1"
        run: |
          pytest backend/test_input_robustness.py backend/test_tool_registry.py backend/test_change_pipeline.py backend/test_cross_modal.py backend/test_query_routing.py -v

      - name: Validate Verified Reports Integrity
        run: |
          python -c "
          import json, glob, os
          reports = glob.glob('reports/*.json')
          assert len(reports) >= 8, f'Expected >=8 report files, found {len(reports)}'
          for r in reports:
              with open(r, 'r', encoding='utf-8') as f:
                  data = json.load(f)
                  assert isinstance(data, dict), f'{r} is not a valid JSON dict'
          print(f'Successfully verified {len(reports)} benchmark reports in reports/')
          "

  frontend-build:
    name: Frontend Build & Type Validation
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Node.js
        uses: actions/setup-node@v4
        with:
          node-version: "20"
          cache: "npm"
          cache-dependency-path: frontend/package-lock.json

      - name: Install frontend dependencies
        working-directory: frontend
        run: npm ci

      - name: Next.js Production Build
        working-directory: frontend
        env:
          NEXT_PUBLIC_API_URL: "http://localhost:8000"
        run: npm run build

```

- **Is `scripts/audit_claims.py` wired into CI?** **YES** (Step `Run Claims & Provenance Auditor` executes `python scripts/audit_claims.py` on line 32).

### 11.4 Mocking and Stubbing in Test Suites
- **`ENABLE_ML_MODELS: "false"` in CI:** In GitHub Actions (`.github/workflows/ci.yml:36`), `ENABLE_ML_MODELS` is explicitly set to `"false"`. This means in automated CI, all deep learning PyTorch checkpoints are bypassed, and all 81 tests run against `fallback_heuristic` classical OpenCV code paths.
- **Visual Evidence Mocking:** Unit tests do not execute end-to-end VLM inference on GPU; instead, they validate that the pipeline emits valid base64 PNG data, GeoJSON geometry dictionaries, and structured error schemas.

## 12. KNOWN GAPS (HONEST AUDIT OF SIH26167 REQUIREMENTS)

The following is an unvarnished audit of requirements from ISRO / SAC Problem Statement SIH26167 that are NOT fully met, partially met, or met only with synthetic/simulated data:

1. **Public Benchmark Datasets Not Downloaded Locally:**
   The authentic multi-gigabyte public benchmarks (RSVQA, VRSBench, CDVQA, and BigEarthNet.txt) are NOT present on disk; only custom Python loader scripts generating 100–120 synthetic sample patches via `np.random` exist in `ml_pipeline/dataset_cache/`.
2. **Model Training on Pseudo-Labels and Synthetic Noise:**
   `SatSegNet` was trained on self-labelled pseudo-masks derived from heuristic OpenCV rules (`satellite_data_engine.py`); `ChangeNet`, `OpticalSARNet`, and `RS-VLM` were trained for only 1 to 3 epochs on 12 to 70 synthetic sample patches.
3. **Report Discrepancy for SatSegNet mIoU:**
   `77.82% mIoU` cited in documentation has no backing JSON report file; the actual evaluation artifact `ml_pipeline/evaluation_results.json` reports `80.22%` on the 30-sample test set, and `training_metrics.json` reports `72.60%` validation mIoU.
4. **Report Discrepancy for ChangeNet mIoU:**
   `71.30% mIoU` cited in documentation has no backing JSON report file; `reports/change.json` reports `15.16%` change mask mIoU on the 15 CDVQA sample pairs.
5. **Report Discrepancy for OpticalSARNet mIoU:**
   `75.80% mIoU` cited in documentation has no backing JSON report file; `reports/cross_modal.json` reports `24.93%` overall mIoU on BigEarthNet sample pairs.
6. **Report Discrepancy for 9.68 ms Latency:**
   `9.68 ms` cited in documentation has no backing entry in `reports/latency.json`; real measured P50 tool latencies range from `7.07 ms` to `229.29 ms`.
7. **Absence of Real Geospatial Multi-Band GeoTIFF Imagery:**
   The repository contains no real georeferenced multi-band GeoTIFF rasters with authentic EPSG projections; only visual 3D textures exist in `frontend/public/textures/`.
8. **Hardcoded Place Name Heuristics:**
   `backend/model_interfaces.py:1279-1280` hardcodes the place name to `"South Mumbai / Western Indian Coastal Harbor & Peninsula Region"` based purely on land/water pixel ratios if georeferencing is missing.
9. **CI Operates Exclusively in Fallback Mode:**
   GitHub Actions CI sets `ENABLE_ML_MODELS: "false"`, testing only the classical OpenCV heuristic fallback routines and never testing the PyTorch neural network checkpoints.
10. **Frontend Type Checking Bypassed:**
    `frontend/next.config.ts` enforces `ignoreBuildErrors: true` to bypass 157 ESLint problems (116 errors, 41 warnings) during deployment builds.
11. **Cross-Modal Raw PNG Ingestion Modality Failure:**
    Uploading raw PNG files (e.g. `risat1_cband_radar_sar.png`) to `/api/query` fails with `MODALITY_MISMATCH` because modality detection requires GeoTIFF radar metadata tags for automatic SAR classification.
12. **Data Leakage in Segmentation Train/Test Split:**
    `satellite_data_engine.py` randomly cropped and augmented patches from the same source images before shuffling them into train, val, and test, meaning overlapping visual content exists across training and evaluation splits.

Wednesday, October 14, 2026 8:16:04 PM 98e155a875fb987d58b74665809fc2d1cfd7b87a