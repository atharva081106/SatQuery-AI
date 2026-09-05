import type { Metadata } from "next";
import "./globals.css";
import CookieConsent from "@/components/CookieConsent";
import { AuthProvider } from "@/context/AuthContext";
import AuthModal from "@/components/AuthModal";

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
        <AuthProvider>
          {children}
          <AuthModal />
          <CookieConsent />
        </AuthProvider>
      </body>
    </html>
  );
}
