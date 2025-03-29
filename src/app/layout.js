import { Inter } from 'next/font/google'
import './globals.css'
import './styles/layout.css'
import './styles/navbar.css'
import './styles/footer.css' 
import './styles/card.css'
import './styles/news.css'  // Adicione esta linha
import Navigation from './components/Navigation'
import Footer from './components/Footer'

const inter = Inter({ subsets: ['latin'] })

export const metadata = {
  title: 'F1 Hub',
  description: 'Your ultimate destination for Formula 1 news and stats',
}

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <div className="layout-container">
          <Navigation />
          <main className="main-container">
            {children}
          </main>
          <Footer />
        </div>
      </body>
    </html>
  )
}