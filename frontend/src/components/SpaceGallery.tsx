"use client";

import React from 'react';
import SphereOrbit from './SphereOrbit';

const galleryImagesData = [
  "https://images.unsplash.com/photo-1451187580459-43490279c0fa?auto=format&fit=crop&w=400&q=80",
  "https://images.unsplash.com/photo-1446776811953-b23d57bd21aa?auto=format&fit=crop&w=400&q=80",
  "https://images.unsplash.com/photo-1462331940025-496dfbfc7564?auto=format&fit=crop&w=400&q=80",
  "https://images.unsplash.com/photo-1541185933-ef5d8ed016c2?auto=format&fit=crop&w=400&q=80",
  "https://images.unsplash.com/photo-1475274047050-1d0c0975c63e?auto=format&fit=crop&w=400&q=80",
  "https://images.unsplash.com/photo-1517976487492-5750f3195933?auto=format&fit=crop&w=400&q=80",
  "https://images.unsplash.com/photo-1419242902214-272b3f66ee7a?auto=format&fit=crop&w=400&q=80",
  "https://images.unsplash.com/photo-1462332420958-a05d1e002413?auto=format&fit=crop&w=400&q=80",
  "https://images.unsplash.com/photo-1506443432602-ac2fcd6f54e0?auto=format&fit=crop&w=400&q=80",
  "https://images.unsplash.com/photo-1537420327992-d6e192287183?auto=format&fit=crop&w=400&q=80",
  "https://images.unsplash.com/photo-1534447677768-be436bb09401?auto=format&fit=crop&w=400&q=80",
  "https://images.unsplash.com/photo-1444703686981-a3abbc4d4fe3?auto=format&fit=crop&w=400&q=80",
  "https://images.unsplash.com/photo-1465101162946-4377e57745c3?auto=format&fit=crop&w=400&q=80",
  "https://images.unsplash.com/photo-1614728263952-84ea256f9679?auto=format&fit=crop&w=400&q=80",
  "https://images.unsplash.com/photo-1614732414444-098e5e111a42?auto=format&fit=crop&w=400&q=80",
  "https://images.unsplash.com/photo-1454789476662-53eb23ba5907?auto=format&fit=crop&w=400&q=80",
  "https://images.unsplash.com/photo-1484589065579-248aad0d8b13?auto=format&fit=crop&w=400&q=80",
  "https://images.unsplash.com/photo-1516339901601-2e1b62dc0c45?auto=format&fit=crop&w=400&q=80",
  "https://images.unsplash.com/photo-1501862700950-18382cd41497?auto=format&fit=crop&w=400&q=80",
  "https://images.unsplash.com/photo-1504333638976-ea0bb97601b6?auto=format&fit=crop&w=400&q=80"
];

export default function SpaceGallery() {
  return (
    <div className="w-full h-screen bg-black overflow-hidden relative flex items-center justify-center">
      <SphereOrbit 
        images={galleryImagesData}
        count={20}
        radius={240}
        tileWidth={100}
        tileHeight={100}
        distance={600}
        speed={14}
        autoRotate={true}
        interactive={true}
        draggable={true}
        background="transparent"
        style={{ width: '100%', height: '100%' }}
      />
    </div>
  );
}
