"use client";

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";

export default function CookieConsent() {
  const [show, setShow] = useState(false);
  const [preferencesOpen, setPreferencesOpen] = useState(false);

  useEffect(() => {
    const consent = localStorage.getItem("cookie_consent");
    if (!consent) {
      const timer = setTimeout(() => setShow(true), 1500);
      return () => clearTimeout(timer);
    }
  }, []);

  const handleAcceptAll = () => {
    localStorage.setItem("cookie_consent", "all");
    setShow(false);
  };

  const handleRejectAll = () => {
    localStorage.setItem("cookie_consent", "necessary");
    setShow(false);
  };

  const handleSavePreferences = () => {
    localStorage.setItem("cookie_consent", "custom");
    setShow(false);
  };

  return (
    <AnimatePresence>
      {show && (
        <motion.div
          initial={{ y: 100, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          exit={{ y: 100, opacity: 0 }}
          transition={{ type: "spring", stiffness: 300, damping: 30 }}
          className="fixed bottom-0 left-0 right-0 md:bottom-6 md:left-1/2 md:-translate-x-1/2 md:w-[500px] z-[99999] px-4 md:px-0"
        >
          <div className="bg-[#0b0c10]/50 backdrop-blur-2xl border border-white/10 p-5 rounded-t-2xl md:rounded-2xl shadow-[0_8px_32px_rgba(0,0,0,0.4)] overflow-hidden relative">
            <div className="absolute inset-0 bg-gradient-to-br from-blue-500/5 to-transparent pointer-events-none" />
            
            <div className="relative z-10">
              {!preferencesOpen ? (
                <>
                  <div className="flex items-start gap-3 mb-4">
                    <div className="p-2 bg-[#1c2230]/80 rounded-full shrink-0">
                      <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-blue-400">
                        <path d="M12 2a10 10 0 1 0 10 10 4 4 0 0 1-5-5 4 4 0 0 1-5-5" />
                        <path d="M8.5 8.5v.01" />
                        <path d="M16 15.5v.01" />
                        <path d="M12 12v.01" />
                        <path d="M11 17v.01" />
                        <path d="M7 14v.01" />
                      </svg>
                    </div>
                    <div>
                      <h3 className="text-lg font-semibold text-white mb-1">We value your privacy</h3>
                      <p className="text-[13px] text-gray-300/90 leading-relaxed">
                        We use cookies to enhance your browsing experience, serve personalized content, and analyze our traffic. By clicking "Accept All", you consent to our use of cookies.
                      </p>
                    </div>
                  </div>
                  
                  <div className="flex gap-3 mt-5">
                    <button
                      onClick={handleAcceptAll}
                      className="flex-1 bg-white text-black font-semibold text-sm py-2 px-4 rounded-xl hover:bg-gray-200 transition-colors active:scale-95"
                    >
                      Accept All
                    </button>
                    <button
                      onClick={handleRejectAll}
                      className="flex-1 bg-[#23252d]/80 text-white font-semibold text-sm py-2 px-4 rounded-xl hover:bg-[#2c2f38] border border-white/5 transition-colors active:scale-95"
                    >
                      Reject All
                    </button>
                  </div>
                  <div className="mt-3 text-center">
                    <button
                      onClick={() => setPreferencesOpen(true)}
                      className="text-xs text-gray-400 hover:text-white transition-colors underline-offset-4 hover:underline"
                    >
                      Manage Preferences
                    </button>
                  </div>
                </>
              ) : (
                <motion.div
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                >
                  <h3 className="text-lg font-semibold text-white mb-3">Cookie Preferences</h3>
                  
                  <div className="space-y-3 mb-5 max-h-[30vh] overflow-y-auto pr-2 custom-scrollbar">
                    <div className="flex items-center justify-between bg-white/5 p-3 rounded-xl border border-white/5">
                      <div>
                        <h4 className="text-white text-sm font-medium">Strictly Necessary</h4>
                        <p className="text-[11px] text-gray-400 mt-0.5">Required for the website to function.</p>
                      </div>
                      <div className="relative shrink-0">
                        <input type="checkbox" className="sr-only" checked disabled />
                        <div className="block bg-blue-500 w-9 h-5 rounded-full opacity-50 cursor-not-allowed"></div>
                        <div className="dot absolute left-1 top-1 bg-white w-3 h-3 rounded-full transition transform translate-x-4"></div>
                      </div>
                    </div>
                    
                    <div className="flex items-center justify-between bg-white/5 p-3 rounded-xl border border-white/5">
                      <div>
                        <h4 className="text-white text-sm font-medium">Analytics</h4>
                        <p className="text-[11px] text-gray-400 mt-0.5">Help us improve by measuring usage.</p>
                      </div>
                      <label className="flex items-center cursor-pointer shrink-0">
                        <div className="relative">
                          <input type="checkbox" className="sr-only peer" defaultChecked />
                          <div className="block bg-gray-600 w-9 h-5 rounded-full peer-checked:bg-blue-500 transition-colors duration-300"></div>
                          <div className="dot absolute left-1 top-1 bg-white w-3 h-3 rounded-full transition-transform duration-300 peer-checked:translate-x-4"></div>
                        </div>
                      </label>
                    </div>
                    
                    <div className="flex items-center justify-between bg-white/5 p-3 rounded-xl border border-white/5">
                      <div>
                        <h4 className="text-white text-sm font-medium">Marketing</h4>
                        <p className="text-[11px] text-gray-400 mt-0.5">Used to deliver relevant ads.</p>
                      </div>
                      <label className="flex items-center cursor-pointer shrink-0">
                        <div className="relative">
                          <input type="checkbox" className="sr-only peer" />
                          <div className="block bg-gray-600 w-9 h-5 rounded-full peer-checked:bg-blue-500 transition-colors duration-300"></div>
                          <div className="dot absolute left-1 top-1 bg-white w-3 h-3 rounded-full transition-transform duration-300 peer-checked:translate-x-4"></div>
                        </div>
                      </label>
                    </div>
                  </div>
                  
                  <div className="flex gap-3 mt-4">
                    <button
                      onClick={handleSavePreferences}
                      className="flex-1 bg-white text-black font-semibold text-sm py-2 px-4 rounded-xl hover:bg-gray-200 transition-colors active:scale-95"
                    >
                      Save Preferences
                    </button>
                    <button
                      onClick={() => setPreferencesOpen(false)}
                      className="bg-[#23252d]/80 text-white font-semibold text-sm py-2 px-4 rounded-xl hover:bg-[#2c2f38] border border-white/5 transition-colors active:scale-95"
                    >
                      Back
                    </button>
                  </div>
                </motion.div>
              )}
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
