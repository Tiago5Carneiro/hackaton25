import { Inter } from 'next/font/google'
import './globals.css'
import './styles/layout.css'
import './styles/navbar.css'
import './styles/footer.css' 
import './styles/card.css'
import './styles/news.css'
import { Titillium_Web } from 'next/font/google';
import Navigation from './components/Navigation'
import Footer from './components/Footer'
import { F1DataProvider } from './context/F1DataContext'

const inter = Inter({ subsets: ['latin'] })

// Initialize the font
const titillium = Titillium_Web({
  subsets: ['latin'],
  display: 'swap',
  weight: ['200', '300', '400', '600', '700', '900'],
  variable: '--font-titillium',
});

export const metadata = {
  title: 'F1 Hub',
  description: 'Your ultimate destination for Formula 1 news and stats',
}

export default function RootLayout({ children }) {
  return (
    <html lang="en" className={titillium.className}>
      <body className={inter.className}>
        <F1DataProvider>
          <div className="layout-container">
            <Navigation />
            <main className="main-container">
              {children}
            </main>
            <Footer />
          </div>
        </F1DataProvider>
      </body>
    </html>
  )
}