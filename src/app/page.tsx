"use client";

import dynamic from "next/dynamic";

const Dashboard = dynamic(() => import("@/components/dashboard"), {
  ssr: false,
  loading: () => (
    <div className="flex min-h-screen items-center justify-center">
      <div className="text-center">
        <div className="mb-4 h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent mx-auto" />
        <p className="text-muted-foreground">Loading dashboard...</p>
      </div>
    </div>
  ),
});

export default function Home() {
  return <Dashboard />;
}
