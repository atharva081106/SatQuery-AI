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
    setPreferencesOpen(false);
  };

  return (
    <AnimatePresence>
      {show && (
        <motion.div
          initial={{ y: 100, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          exit={{ y: 100, opacity: 0 }}
          transition={{ type: "spring", stiffness: 300, damping: 30 }}
          className="fixed bottom-0 left-0 right-0 z-[99999]"
        >
          {/* Main Single Line Footer */}
          <div className="bg-[#0b0c10]/70 backdrop-blur-2xl border-t border-white/10 px-4 py-3 shadow-[0_-8px_32px_rgba(0,0,0,0.4)]">
            <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
              <div className="flex items-center gap-3">
                <div className="p-1.5 bg-[#1c2230]/80 rounded-full shrink-0 hidden sm:block">
                  <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-blue-400">
                    <path d="M12 2a10 10 0 1 0 10 10 4 4 0 0 1-5-5 4 4 0 0 1-5-5" />
                    <path d="M8.5 8.5v.01" />
                    <path d="M16 15.5v.01" />
                    <path d="M12 12v.01" />
                    <path d="M11 17v.01" />
                    <path d="M7 14v.01" />
                  </svg>
                </div>
                <p className="text-[12px] md:text-[13px] text-gray-300/90 leading-tight text-center md:text-left">
                  We use cookies to enhance your experience. By continuing, you agree to our use of cookies.
                </p>
              </div>
              
              <div className="flex items-center gap-3 shrink-0">
                <button
                  onClick={() => setPreferencesOpen(!preferencesOpen)}
                  className="text-[11px] md:text-xs text-gray-400 hover:text-white transition-colors underline-offset-4 hover:underline mr-1"
                >
                  Preferences
                </button>
                <button
                  onClick={handleRejectAll}
                  className="bg-[#23252d]/80 text-white font-medium text-[11px] md:text-xs py-1.5 px-4 rounded-full hover:bg-[#2c2f38] border border-white/5 transition-colors active:scale-95"
                >
                  Reject All
                </button>
                <button
                  onClick={handleAcceptAll}
                  className="bg-white text-black font-semibold text-[11px] md:text-xs py-1.5 px-4 rounded-full hover:bg-gray-200 transition-colors active:scale-95"
                >
                  Accept All
                </button>
              </div>
            </div>
          </div>

          {/* Preferences Floating Modal (Appears above the footer) */}
          <AnimatePresence>
            {preferencesOpen && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 20 }}
                className="absolute bottom-full left-0 right-0 md:left-auto md:right-4 mb-4 p-4 md:w-[380px]"
              >
                <div className="bg-[#0b0c10]/90 backdrop-blur-2xl border border-white/10 p-5 rounded-2xl shadow-[0_8px_32px_rgba(0,0,0,0.6)]">
                  <h3 className="text-[15px] font-semibold text-white mb-3">Cookie Preferences</h3>
                  
                  <div className="space-y-2.5 mb-4 max-h-[30vh] overflow-y-auto pr-2 custom-scrollbar">
                    <div className="flex items-center justify-between bg-white/5 p-2.5 rounded-lg border border-white/5">
                      <div>
                        <h4 className="text-white text-[13px] font-medium">Strictly Necessary</h4>
                        <p className="text-[10px] text-gray-400 mt-0.5">Required for the website to function.</p>
                      </div>
                      <div className="relative shrink-0">
                        <input type="checkbox" className="sr-only" checked disabled />
                        <div className="block bg-blue-500 w-8 h-4 rounded-full opacity-50 cursor-not-allowed"></div>
                        <div className="dot absolute left-0.5 top-0.5 bg-white w-3 h-3 rounded-full transition transform translate-x-4"></div>
                      </div>
                    </div>
                    
                    <div className="flex items-center justify-between bg-white/5 p-2.5 rounded-lg border border-white/5">
                      <div>
                        <h4 className="text-white text-[13px] font-medium">Analytics</h4>
                        <p className="text-[10px] text-gray-400 mt-0.5">Help us improve by measuring usage.</p>
                      </div>
                      <label className="flex items-center cursor-pointer shrink-0">
                        <div className="relative">
                          <input type="checkbox" className="sr-only peer" defaultChecked />
                          <div className="block bg-gray-600 w-8 h-4 rounded-full peer-checked:bg-blue-500 transition-colors duration-300"></div>
                          <div className="dot absolute left-0.5 top-0.5 bg-white w-3 h-3 rounded-full transition-transform duration-300 peer-checked:translate-x-4"></div>
                        </div>
                      </label>
                    </div>
                  </div>
                  
                  <button
                    onClick={handleSavePreferences}
                    className="w-full bg-white text-black font-semibold text-[13px] py-2 rounded-xl hover:bg-gray-200 transition-colors active:scale-95"
                  >
                    Save Preferences
                  </button>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
