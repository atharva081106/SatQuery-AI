import type { Metadata } from "next";
import "./globals.css";
import CookieConsent from "@/components/CookieConsent";
import { AuthProvider } from "@/context/AuthContext";
import AuthModal from "@/components/AuthModal";
import SmoothScroll from "@/components/SmoothScroll";

import Script from "next/script";

export const metadata: Metadata = {
  title: "SatQuery AI - Earth Observation Intelligence",
  description: "Multimodal Remote Sensing Image Analysis through Natural Language Queries",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>
        <Script src="https://accounts.google.com/gsi/client" strategy="lazyOnload" />
        <SmoothScroll>
          <AuthProvider>
            {children}
            <AuthModal />
            <CookieConsent />
          </AuthProvider>
        </SmoothScroll>
      </body>
    </html>
  );
}

