"use client";

import React, { createContext, useContext, useState, useEffect } from "react";

export interface User {
  name: string;
  email: string;
  role?: string;
  isJudge?: boolean;
  provider?: "google" | "email" | "judge";
}

interface AuthContextType {
  user: User | null;
  isAuthenticated: boolean;
  queryCount: number;
  mapCount: number;
  maxFreeQueries: number;
  maxFreeMapQueries: number;
  canQuery: boolean;
  canUseMap: boolean;
  isAuthModalOpen: boolean;
  authModalReason: string;
  authModalInitialMode?: "signin" | "signup";
  openAuthModal: (reason?: string, initialMode?: "signin" | "signup") => void;
  closeAuthModal: () => void;
  incrementQueryCount: () => void;
  incrementMapCount: () => void;
  login: (email: string, name?: string) => void;
  signup: (name: string, email: string) => void;
  loginWithGoogle: (email?: string, name?: string) => void;
  loginAsJudge: () => void;
  logout: () => void;
}


const AuthContext = createContext<AuthContextType | undefined>(undefined);

const MAX_FREE_QUERIES = 3;
const MAX_FREE_MAP_QUERIES = 1;

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [queryCount, setQueryCount] = useState<number>(0);
  const [mapCount, setMapCount] = useState<number>(0);
  const [isAuthModalOpen, setIsAuthModalOpen] = useState<boolean>(false);
  const [authModalReason, setAuthModalReason] = useState<string>("");
  const [authModalInitialMode, setAuthModalInitialMode] = useState<"signin" | "signup">("signin");

  useEffect(() => {
    // Hydrate state from localStorage
    try {
      const savedUser = localStorage.getItem("satquery_user");
      if (savedUser) {
        setUser(JSON.parse(savedUser));
      }

      const savedQueryCount = localStorage.getItem("satquery_query_count");
      if (savedQueryCount) {
        setQueryCount(parseInt(savedQueryCount, 10) || 0);
      }

      const savedMapCount = localStorage.getItem("satquery_map_count");
      if (savedMapCount) {
        setMapCount(parseInt(savedMapCount, 10) || 0);
      }
    } catch (e) {
      console.warn("Could not read auth/quota from localStorage", e);
    }
  }, []);

  const openAuthModal = (reason?: string, initialMode?: "signin" | "signup") => {
    setAuthModalReason(
      reason || "Authentication required to access advanced satellite analytics."
    );
    if (initialMode) {
      setAuthModalInitialMode(initialMode);
    }
    setIsAuthModalOpen(true);
  };

  const closeAuthModal = () => {
    setIsAuthModalOpen(false);
  };

  const incrementQueryCount = () => {
    if (user) return; // Unlimited for authenticated users
    setQueryCount((prev) => {
      const next = prev + 1;
      try {
        localStorage.setItem("satquery_query_count", next.toString());
      } catch (e) {}
      return next;
    });
  };

  const incrementMapCount = () => {
    if (user) return; // Unlimited for authenticated users
    setMapCount((prev) => {
      const next = prev + 1;
      try {
        localStorage.setItem("satquery_map_count", next.toString());
      } catch (e) {}
      return next;
    });
  };

  const login = (email: string, name?: string) => {
    const newUser: User = {
      name: name || email.split("@")[0] || "Operator",
      email,
      role: "OPERATOR",
    };
    setUser(newUser);
    try {
      localStorage.setItem("satquery_user", JSON.stringify(newUser));
    } catch (e) {}
    setIsAuthModalOpen(false);
  };

  const signup = (name: string, email: string) => {
    const newUser: User = {
      name,
      email,
      role: "OPERATOR",
    };
    setUser(newUser);
    try {
      localStorage.setItem("satquery_user", JSON.stringify(newUser));
    } catch (e) {}
    setIsAuthModalOpen(false);
  };

  const loginAsJudge = () => {
    const judgeUser: User = {
      name: "SIH Technical Evaluator",
      email: "evaluator@isro-sih.gov.in",
      role: "EVALUATOR_PASS",
      isJudge: true,
    };
    setUser(judgeUser);
    try {
      localStorage.setItem("satquery_user", JSON.stringify(judgeUser));
    } catch (e) {}
    setIsAuthModalOpen(false);
  };

  const loginWithGoogle = (email?: string, name?: string) => {
    const googleUser: User = {
      name: name || "Google Verified Operator",
      email: email || "operator.isro@gmail.com",
      role: "GOOGLE_VERIFIED",
      provider: "google",
    };
    setUser(googleUser);
    try {
      localStorage.setItem("satquery_user", JSON.stringify(googleUser));
    } catch (e) {}
    setIsAuthModalOpen(false);
  };

  const logout = () => {
    setUser(null);
    try {
      localStorage.removeItem("satquery_user");
    } catch (e) {}
  };


  const isAuthenticated = !!user;
  const canQuery = isAuthenticated || queryCount < MAX_FREE_QUERIES;
  const canUseMap = isAuthenticated || mapCount < MAX_FREE_MAP_QUERIES;

  return (
    <AuthContext.Provider
      value={{
        user,
        isAuthenticated,
        queryCount,
        mapCount,
        maxFreeQueries: MAX_FREE_QUERIES,
        maxFreeMapQueries: MAX_FREE_MAP_QUERIES,
        canQuery,
        canUseMap,
        isAuthModalOpen,
        authModalReason,
        authModalInitialMode,
        openAuthModal,
        closeAuthModal,
        incrementQueryCount,
        incrementMapCount,
        login,
        signup,
        loginWithGoogle,
        loginAsJudge,
        logout,
      }}

    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
}
