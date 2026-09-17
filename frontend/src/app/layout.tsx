import type { Metadata, Viewport } from "next";
import "./globals.css";
import CookieConsent from "@/components/CookieConsent";
import { AuthProvider } from "@/context/AuthContext";
import AuthModal from "@/components/AuthModal";
import SmoothScroll from "@/components/SmoothScroll";
import PWAController from "@/components/PWAController";

import Script from "next/script";

export const metadata: Metadata = {
  title: "SatQuery AI - Earth Observation Intelligence",
  description: "Multimodal Remote Sensing Image Analysis through Natural Language Queries",
  applicationName: "SatQuery AI",
  manifest: "/manifest.json",
  appleWebApp: {
    capable: true,
    statusBarStyle: "black-translucent",
    title: "SatQuery AI",
  },
  formatDetection: {
    telephone: false,
  },
  icons: {
    icon: [
      { url: "/icons/icon.svg", type: "image/svg+xml" },
      { url: "/icons/icon-192x192.png", sizes: "192x192", type: "image/png" },
      { url: "/icons/icon-512x512.png", sizes: "512x512", type: "image/png" },
    ],
    apple: [
      { url: "/icons/apple-touch-icon.png", sizes: "180x180", type: "image/png" },
    ],
  },
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 5,
  userScalable: true,
  viewportFit: "cover",
  themeColor: "#030712",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        <meta name="apple-mobile-web-app-capable" content="yes" />
        <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent" />
        <link rel="apple-touch-icon" href="/icons/apple-touch-icon.png" />
      </head>
      <body>
        <Script src="https://accounts.google.com/gsi/client" strategy="beforeInteractive" />
        <SmoothScroll>
          <AuthProvider>
            {children}
            <AuthModal />
            <CookieConsent />
            <PWAController />
          </AuthProvider>
        </SmoothScroll>
      </body>
    </html>
  );
}

