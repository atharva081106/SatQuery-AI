import type { Metadata } from "next";
import "./globals.css";
import CookieConsent from "@/components/CookieConsent";
import CometCursor from "@/components/CometCursor";

export const metadata: Metadata = {
  title: "SatQuery AI",
  description: "Multimodal Remote Sensing Image Analysis through Text Queries",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>
        {children}
        <CookieConsent />
        <CometCursor 
          variant="nebula" 
          trailColor="#00f0ff" 
          coreColor="#ffffff" 
          hideCursor={true}
          layer={99999}
        />
      </body>
    </html>
  );
}
